import json
import os

# 实体类型映射
entity_map = {
    "name": {"instruction": "人物实体，人物实体指的是句子中提到的人名。", "label": "人物"},
    "company": {"instruction": "公司实体，公司实体指的是句子中提到的公司名称。", "label": "公司"},
    "organization": {"instruction": "组织实体，组织实体指的是句子中提到的组织名称。", "label": "组织"},
    "address": {"instruction": "地址实体，地址实体指的是句子中提到的地点名称。", "label": "地点"},
    "position": {"instruction": "职位实体，职位实体指的是句子中提到的职位名称。", "label": "职位"},
    "game": {"instruction": "游戏实体，游戏实体指的是句子中提到的游戏名称。", "label": "游戏"},
    "movie": {"instruction": "电影实体，电影实体指的是句子中提到的电影名称。", "label": "电影"},
    "book": {"instruction": "书籍实体，书籍实体指的是句子中提到的书籍名称。", "label": "书籍"},
    "government": {"instruction": "政府机构实体，政府机构实体指的是句子中提到的政府机构名称。", "label": "政府"},
    "scene": {"instruction": "场景实体，场景实体指的是句子中提到的场景名称。", "label": "场景"}
}

def convert_cluener_to_baner(input_file, output_file):
    """将CLUENER格式转换为BANER格式"""
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
        
        # 为每种实体类型生成一条数据
        for entity_type, entities in label.items():
            if entity_type not in entity_map:
                continue
            
            # 提取该类型的所有实体
            entity_list = []
            for entity_name, positions in entities.items():
                # 检查实体名称是否在文本中
                if entity_name in text:
                    entity_list.append(entity_name)
            
            if entity_list:
                # 生成输出格式
                output = []
                for entity_name in entity_list:
                    output.append(f"<<<{entity_name}>>>{entity_map[entity_type]['label']}")
                output_str = ''.join(output) + '<im_end>'
                
                # 生成指令
                instruction = f"请提取输入句子中的{entity_map[entity_type]['instruction']}"
                
                # 生成BANER格式数据
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
    
    print(f"转换完成，生成了{len(converted_data)}条数据")

if __name__ == "__main__":
    input_file = "data/cluener.json"
    output_file = "data/cluener_converted.json"
    convert_cluener_to_baner(input_file, output_file)