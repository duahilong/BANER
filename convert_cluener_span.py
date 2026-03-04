import json
import os

def convert_cluener_to_span(input_file, output_file):
    """将CLUENER格式转换为BANER Stage-1所需的片段级格式（不含实体类型）"""
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    converted_data = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        
        text = data.get('text', '')
        label = data.get('label', {})
        
        # 提取所有实体片段（不含类型）
        entity_spans = []
        for entity_type, entities in label.items():
            for entity_name, positions in entities.items():
                if entity_name in text:
                    entity_spans.append(entity_name)
        
        # 去重
        entity_spans = list(set(entity_spans))
        
        if entity_spans:
            # 生成Stage-1格式的输出：实体片段及其位置
            # 输出格式：片段1 (起始-结束), 片段2 (起始-结束), ...
            output_parts = []
            for span in entity_spans:
                start = text.find(span)
                if start != -1:
                    end = start + len(span) - 1
                    output_parts.append(f"{span} ({start}-{end})")
            
            output_str = ', '.join(output_parts) + '<im_end>'
            
            # 使用Stage-1的中文指令
            instruction = "请忽略实体类型，仅关注文本中所有潜在的命名实体片段。将识别出的片段以结构化索引 (起始和结束位置) 形式输出。"
            
            baner_data = {
                "instruction": instruction,
                "input": text,
                "output": output_str
            }
            converted_data.append(baner_data)
    
    # 保存转换后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"转换完成，生成了{len(converted_data)}条Stage-1训练数据")

if __name__ == "__main__":
    input_file = "data/cluener.json"
    output_file = "data/cluener_stage1.json"
    convert_cluener_to_span(input_file, output_file)