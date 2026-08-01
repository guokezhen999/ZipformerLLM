import json
import jiwer
import argparse
import os
import re
import unicodedata
from datetime import datetime

def get_display_width(s):
    """计算字符串的终端显示宽度（中文字符算2个宽度，英文字符算1个宽度）"""
    width = 0
    for c in s:
        if unicodedata.east_asian_width(c) in ('W', 'F'):
            width += 2
        else:
            width += 1
    return width

def pad_string(s, target_width):
    """根据终端显示宽度进行左对齐填充"""
    current_width = get_display_width(s)
    padding = max(0, target_width - current_width)
    return s + " " * padding

def normalize_text(text, lang):
    """
    多语言清洗逻辑：
    - en/es: 转大写，移除标点，保留字母和数字，压缩空格 (用于计算 WER)
    - zh   : 转大写，移除标点，去除所有原空格后，在每个字符间插入空格 (用于计算 CER)
    """
    if not text:
        return ""
    
    text = str(text).upper()
    text = re.sub(r'[^\w\s]|_', '', text)
    
    if lang == 'zh':
        text = text.replace(" ", "")
        text = " ".join(list(text))
    else:
        text = " ".join(text.split())
        
    return text

def generate_alignment_visual(output):
    """
    根据 jiwer 的 output 对象生成三行对齐的字符串
    """
    alignment = output.alignments[0]
    ref_tokens = output.references[0]
    hyp_tokens = output.hypotheses[0]
    
    line_ref = []
    line_hyp = []
    line_op = []

    for chunk in alignment:
        op_type = chunk.type
        r_start, r_end = chunk.ref_start_idx, chunk.ref_end_idx
        h_start, h_end = chunk.hyp_start_idx, chunk.hyp_end_idx
        
        ref_segment = ref_tokens[r_start:r_end]
        hyp_segment = hyp_tokens[h_start:h_end]
        
        max_len = max(len(ref_segment), len(hyp_segment))
        
        for i in range(max_len):
            txt_ref = ref_segment[i] if i < len(ref_segment) else "***"
            txt_hyp = hyp_segment[i] if i < len(hyp_segment) else "***"
            
            if op_type == 'equal':
                op_code = ' '   
            elif op_type == 'substitute':
                op_code = 'S'   
            elif op_type == 'delete':
                op_code = 'D'   
            elif op_type == 'insert':
                op_code = 'I'   
            else:
                op_code = '?'

            w_ref = get_display_width(txt_ref)
            w_hyp = get_display_width(txt_hyp)
            w_op = get_display_width(op_code)
            
            col_width = max(w_ref, w_hyp, w_op) + 2
            
            line_ref.append(pad_string(txt_ref, col_width))
            line_hyp.append(pad_string(txt_hyp, col_width))
            line_op.append(pad_string(op_code, col_width))
            
    return "".join(line_ref), "".join(line_hyp), "".join(line_op)

def load_ground_truths(filepath):
    """解析原始文件，提取 id 和 ground truth 文本"""
    ground_truths = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.strip(): continue
            try:
                data = json.loads(line)
                uid = data.get("id")
                
                # 从 supervisions 中提取真实的文本
                text = ""
                supervisions = data.get("supervisions", [])
                if supervisions and isinstance(supervisions, list):
                    text = supervisions[0].get("text", "")
                
                if uid and text:
                    ground_truths[uid] = text
            except Exception as e:
                print(f"解析真实标签文件第 {i+1} 行出错: {e}")
    return ground_truths

def load_predictions(filepath):
    """解析流式推理结果文件，提取 id 和合并后的 prediction 文本"""
    predictions = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.strip(): continue
            try:
                data = json.loads(line)
                uid = data.get("id")
                
                # 将 segments_text 数组拼成一个完整的字符串
                segments = data.get("segments_text", [])
                if segments and isinstance(segments, list):
                    text = " ".join([seg for seg in segments if isinstance(seg, str) and seg.strip()])
                else:
                    # 兼容非流式或常规格式 (如 text 或 prediction 字段)
                    text = data.get("text", "") or data.get("prediction", "") or ""
                    text = str(text).strip()
                
                if uid:
                    predictions[uid] = text
            except Exception as e:
                print(f"解析预测文件第 {i+1} 行出错: {e}")
    return predictions

def main():
    parser = argparse.ArgumentParser(description="合并流式推理结果并生成 WER/CER 详细统计")
    parser.add_argument("-r", "--ref", required=True, help="原始标签文件路径 (包含 supervisions)")
    parser.add_argument("-p", "--pred", required=True, help="流式推理结果文件路径 (包含 segments_text)")
    parser.add_argument("-o", "--output", required=True, help="结果报告保存的路径")
    parser.add_argument("-l", "--lang", choices=['en', 'es', 'zh'], default='en', 
                        help="评估语言。en: 英文(WER), es: 西语(WER), zh: 中文(CER)")
    
    args = parser.parse_args()

    if not os.path.exists(args.ref) or not os.path.exists(args.pred):
        print("错误: 找不到输入的文件，请检查路径。")
        return

    metric_name = "CER" if args.lang == 'zh' else "WER"
    print(f"正在加载并合并文件... (语言: {args.lang.upper()}, 指标: {metric_name})")

    # 1. 加载并合并数据
    ground_truths = load_ground_truths(args.ref)
    predictions = load_predictions(args.pred)
    
    # 找出两份文件中都有的 ID (取交集)
    common_ids = [uid for uid in ground_truths if uid in predictions]
    print(f"标签总数: {len(ground_truths)} | 预测总数: {len(predictions)} | 匹配成功的记录数: {len(common_ids)}")

    if not common_ids:
        print("错误: 两个文件之间没有匹配的 ID，无法计算。")
        return

    processed_data = []
    clean_refs = []
    clean_hyps = []
    skipped_count = 0

    # 2. 计算 WER 并生成对齐结果
    for uid in common_ids:
        raw_gt = ground_truths[uid]
        raw_pred = predictions.get(uid, "")
        
        gt_clean = normalize_text(raw_gt, args.lang)
        pred_clean = normalize_text(raw_pred, args.lang)

        if not gt_clean:
            skipped_count += 1
            continue
        
        if not pred_clean:
            pred_clean = " "

        try:
            output = jiwer.process_words(gt_clean, pred_clean)
            vis_ref, vis_hyp, vis_op = generate_alignment_visual(output)
            
            processed_data.append({
                "id": uid,
                "error_rate": output.wer,
                "hits": output.hits,
                "subs": output.substitutions,
                "dels": output.deletions,
                "ins": output.insertions,
                "vis_ref": vis_ref,
                "vis_hyp": vis_hyp,
                "vis_op": vis_op
            })
            
            clean_refs.append(gt_clean)
            clean_hyps.append(pred_clean)
        except Exception as e:
            print(f"处理 ID {uid} 时出错: {e}")
            skipped_count += 1

    # 3. 生成报告
    global_output = jiwer.process_words(clean_refs, clean_hyps)

    report = []
    report.append("="*80)
    report.append(f"语音识别 {metric_name} 深度分析报告 (流式合并版) - 语言: {args.lang.upper()}")
    report.append(f"生成日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("-" * 80)
    report.append(f"总体 {metric_name.ljust(8)}: {global_output.wer:.4f} ({global_output.wer * 100:.2f}%)")
    report.append(f"正确 (Hits)   : {global_output.hits}")
    report.append(f"替换 (Subs)   : {global_output.substitutions}")
    report.append(f"删除 (Dels)   : {global_output.deletions}")
    report.append(f"插入 (Ins)    : {global_output.insertions}")
    if skipped_count > 0:
        report.append(f"跳过空行/错误: {skipped_count} 条")
    report.append("="*80)
    report.append("错误详情图例: S=替换, D=删除, I=插入 (空格为正确)")
    report.append("REF = 原句(Reference), HYP = 预测(Hypothesis), OP = 错误类型")
    report.append("="*80)
    
    for d in processed_data:
        stats_str = f"{metric_name}: {d['error_rate']:.4f} | H:{d['hits']} S:{d['subs']} D:{d['dels']} I:{d['ins']}"
        report.append(f"\nID: {d['id']}")
        report.append(f"Stats: {stats_str}")
        report.append("-" * 40)
        report.append(f"REF: {d['vis_ref']}")
        report.append(f"HYP: {d['vis_hyp']}")
        report.append(f"OP : {d['vis_op']}")
        report.append("." * 80)

    with open(args.output, 'w', encoding='utf-8') as f:
        f.write("\n".join(report))
    
    print(f"\n[计算完成] 总体 {metric_name}: {global_output.wer:.4f}")
    print(f"详细对齐报告已保存至: {args.output}")

if __name__ == "__main__":
    main()