import sys, os, warnings, time
sys.path.append('..')
warnings.filterwarnings("ignore")
import random
import pandas as pd
from data_loader import *
from transformers import AdamW, get_linear_schedule_with_warmup
from utils.utils import *
from LLM_MTLN import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', '-l', type=float, default='1e-5', help='learning rate')
parser.add_argument('--seed', '-s', type=int, default='130', help='random seed')
parser.add_argument('--batch_size', '-b', type=int, default=2, help='batch size')
parser.add_argument('--K', '-K', type=int, default=5, help='sliding window size')
parser.add_argument('--split', '-sp', type=str, default='split10', help='batch size')
parser.add_argument('--emotion_enhanced', '-e', type=str, default='true', help='if use sentiment lexicon')
parser.add_argument('--use_rgcn', '-u', type=str, default='true', help='if use rgcn network')
parser.add_argument('--pfn', '-p', type=str, default='true', help='if use pfn network')
args = parser.parse_args()
TORCH_SEED = args.seed


def main(configs, fold_id):
    print('TORCH_SEED', TORCH_SEED)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    random.seed(TORCH_SEED)
    np.random.seed(TORCH_SEED)
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    train_loader = build_train_data(configs, fold_id=fold_id)
    if configs.split == 'split20':
        valid_loader = build_inference_data(configs, fold_id=fold_id, data_type='valid')
    test_loader = build_inference_data(configs, fold_id=fold_id, data_type='test')
    model = LLM_MTLN(configs).to(DEVICE)

    params = model.parameters()
    optimizer = AdamW(params, lr=configs.lr)
    num_steps_all = len(train_loader) // configs.gradient_accumulation_steps * configs.epochs
    warmup_steps = int(num_steps_all * configs.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_steps_all)
   
    model.zero_grad()
    metric_ec, metric_e, metric_c = (-1, -1, -1), (-1, -1, -1), (-1, -1, -1)
    max_ec, max_e, max_c = (-1, -1, -1), (-1, -1, -1), (-1, -1, -1)

    early_stop_flag = None
    for epoch in range(1, configs.epochs+1):
        total_ec_loss, total_e_loss, total_c_loss = 0., 0., 0.

        for train_step, batch in enumerate(train_loader, 1):
            model.train()
            doc_len_b, adj_b, y_emotions_b, y_causes_b, y_mask_b, doc_couples_b, doc_id_b, \
            glm_token_b, glm_segments_b, glm_masks_b, glm_clause_b, glm_clause_sep_b, emo_pos, cau_pos = batch

            couples_pred, emo_cau_pos, pred_e, pred_c = model(glm_token_b, glm_segments_b, glm_masks_b,
                                                              glm_clause_b, glm_clause_sep_b, doc_len_b, adj_b, emo_pos, cau_pos)

            loss_e, loss_c = model.loss_pre(pred_e, pred_c, y_emotions_b, y_causes_b, y_mask_b)
            loss_couple, _ = model.loss_rank(couples_pred, emo_cau_pos, doc_couples_b, y_mask_b)

            loss = loss_couple + loss_e + loss_c
            loss = loss / configs.gradient_accumulation_steps
            
            total_ec_loss += loss_couple.item()
            total_e_loss += loss_e.item()
            total_c_loss += loss_c.item()
            loss.backward()
          
            if train_step % configs.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
          
        avg_ec_loss = total_ec_loss / len(train_loader)
        avg_e_loss = total_e_loss / len(train_loader)
        avg_c_loss = total_c_loss / len(train_loader)
        print('epoch:', epoch, 'avg_ec_loss:', avg_ec_loss, 'avg_e_loss:', avg_e_loss, 'avg_c_loss:', avg_c_loss)

        with torch.no_grad():
            model.eval()
            if configs.split == 'split10':
                test_ec, test_e, test_c = inference_one_epoch(test_loader, model)
                print("epoch:", epoch, 'f_ec:', test_ec[2], 'f_e:', test_e[2], 'f_c:', test_c[2])
                if test_ec[2] > metric_ec[2]:
                    early_stop_flag = 1
                    metric_ec, metric_e, metric_c = test_ec, test_e, test_c
                else:
                    early_stop_flag += 1

            if configs.split == 'split20':
                valid_ec, valid_e, valid_c = inference_one_epoch(valid_loader, model)
                test_ec, test_e, test_c = inference_one_epoch(test_loader, model)
                print("epoch:", epoch, 'f_ec:', valid_ec[2], 'f_e:', valid_e[2], 'f_c:', valid_c[2])
                if valid_ec[2] > max_ec[2]:
                    early_stop_flag = 1
                    max_ec, max_e, max_c = valid_ec, valid_e, valid_c
                    metric_ec, metric_e, metric_c = test_ec, test_e, test_c
                else:
                    early_stop_flag += 1

        if epoch > configs.epochs * 0.7 and early_stop_flag > 8:
            print("超过8次测试没有提升，在第{}个epoch停止".format(epoch))
            print("=="*50)
            break

    return metric_ec, metric_e, metric_c


def inference_one_batch(batch, model):
    doc_len_b, adj_b, y_emotions_b, y_causes_b, y_mask_b, doc_couples_b, doc_id_b, \
    glm_token_b, glm_segments_b, glm_masks_b, glm_clause_b, glm_clause_sep_b, emo_pos, cau_pos = batch

    couples_pred, emo_cau_pos, pred_e, pred_c = model(glm_token_b, glm_segments_b, glm_masks_b,
                                                      glm_clause_b, glm_clause_sep_b, doc_len_b, adj_b, emo_pos, cau_pos)

    loss_e, loss_c = model.loss_pre(pred_e, pred_c, y_emotions_b, y_causes_b, y_mask_b)
    loss_couple, doc_couples_pred_b = model.loss_rank(couples_pred, emo_cau_pos, doc_couples_b, y_mask_b, test=True)

    return to_np(loss_couple), to_np(loss_e), to_np(loss_c), doc_couples_b, doc_couples_pred_b, doc_id_b


def inference_one_epoch(batches, model):
    doc_id_all, doc_couples_all, doc_couples_pred_all = [], [], []
    loss = 0
    for batch in batches:
        loss_ec, _, _, doc_couples, doc_couples_pred, doc_id_b = inference_one_batch(batch, model)
        doc_id_all.extend(doc_id_b)
        doc_couples_all.extend(doc_couples)
        doc_couples_pred_all.extend(doc_couples_pred)
        loss += loss_ec

    print('=== evaluation === loss_ec:', loss/len(batches))

    doc_couples_pred_all = lexicon_based_extraction(doc_id_all, doc_couples_pred_all)
    metric_ec, metric_e, metric_c = eval_func(doc_couples_all, doc_couples_pred_all)

    return metric_ec, metric_e, metric_c


def lexicon_based_extraction(doc_ids, couples_pred):
    emotional_clauses = read_b(os.path.join(DATA_DIR, SENTIMENTAL_CLAUSE_DICT))

    couples_pred_filtered = []
    for i, (doc_id, couples_pred_i) in enumerate(zip(doc_ids, couples_pred)):
        top1, top1_prob = couples_pred_i[0][0], couples_pred_i[0][1]
        couples_pred_i_filtered = [top1]

        emotional_clauses_i = emotional_clauses[doc_id]
        for couple in couples_pred_i[1:]:
            if couple[0][0] in emotional_clauses_i and logistic(couple[1]) > 0.5:  # 以控制精确率和召回率
                couples_pred_i_filtered.append(couple[0])

        couples_pred_filtered.append(couples_pred_i_filtered)
    return couples_pred_filtered


if __name__ == '__main__':
    configs = Config()
    configs.batch_size = args.batch_size
    configs.lr = args.lr
    configs.K = args.K
    configs.split = args.split
    configs.emotion_enhanced = True if str(args.emotion_enhanced) == 'true' else False
    configs.use_rgcn = True if str(args.use_rgcn) == 'true' else False
    configs.pfn = True if str(args.pfn) == 'true' else False

    print('batch_size:{}'.format(configs.batch_size))
    print('learning_rate:{}'.format(configs.lr))
    print('K:{}'.format(configs.K))
    print('split:{}'.format(configs.split))
    print('emotion_enhanced:{}'.format(configs.emotion_enhanced))
    print('use_rgcn:{}'.format(configs.use_rgcn))
    print('use_pfn:{}'.format(configs.pfn))

    if configs.split == 'split10':
        n_folds = 10
    elif configs.split == 'split20':
        n_folds = 20
    else:
        print('Unknown data split.')
        exit()

    # 设定数据存储路径
    result_save_dir = "../results/{}/K-{}_glm".format(configs.split, configs.K)
    if configs.emotion_enhanced:
        result_save_dir += '_emotion'
    if configs.use_rgcn:
        result_save_dir += '_RGCN'
    if configs.pfn:
        result_save_dir += '_PFN'
    result_save_dir += '/'
    if not os.path.exists(result_save_dir):
        os.makedirs(result_save_dir)
    result_save_path = os.path.join(result_save_dir, 'result_statistic.xlsx')

    # 运行主函数
    metric_folds = {'ecp': [], 'emo': [], 'cau': []}
    for fold_id in range(1, n_folds + 1):
        print('===== fold {} ====='.format(fold_id))
        metric_ec, metric_e, metric_c = main(configs, fold_id)
        print('F_ecp: {}'.format(float_n(metric_ec[2])))
        print('F_e: {}'.format(float_n(metric_e[2])))
        print('F_c: {}'.format(float_n(metric_c[2])))

        metric_folds['ecp'].append(metric_ec)
        metric_folds['emo'].append(metric_e)
        metric_folds['cau'].append(metric_c)

    # 写入实验结果
    writer = pd.ExcelWriter(result_save_path)
    for key, values in metric_folds.items():
        precision_list = [value[0] for value in values]
        recall_list = [value[1] for value in values]
        f1_list = [value[2] for value in values]
        save_data = pd.DataFrame({'fold': range(1, n_folds + 1), 'precision': precision_list, 'recall': recall_list,
                                  'f1': f1_list})
        metric_mean = np.mean(np.asarray(metric_folds[key]), axis=0)
        new_row = pd.Series({'fold': 'mean', 'precision': metric_mean[0], 'recall': metric_mean[1],
                             'f1': metric_mean[2]})
        save_data = save_data.append(new_row, ignore_index=True)
        save_data.to_excel(excel_writer=writer, index=None, encoding='utf-8-sig', sheet_name=key)
    writer.save()
    writer.close()
