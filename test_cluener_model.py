import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型和tokenizer
base_model = "./mods/Qwen3.5-9B"
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# 测试句子
test_sentences = [
    "浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。",
    "生生不息CSOL生化狂潮让你填弹狂扫",
    "那不勒斯vs锡耶纳以及桑普vs热那亚之上呢？",
    "布鲁京斯研究所桑顿中国中心研究部主任李成说，东亚的和平与安全，是美国的核心利益之一。",
    "目前主赞助商暂时空缺，他们的球衣上印的是unicef（联合国儿童基金会），是公益性质的广告。"
]

# 实体类型和指令映射
entity_types = {
    "人物": "请提取输入句子中的人物实体，人物实体指的是句子中提到的人名。",
    "公司": "请提取输入句子中的公司实体，公司实体指的是句子中提到的公司名称。",
    "组织": "请提取输入句子中的组织实体，组织实体指的是句子中提到的组织名称。",
    "地点": "请提取输入句子中的地址实体，地址实体指的是句子中提到的地点名称。",
    "职位": "请提取输入句子中的职位实体，职位实体指的是句子中提到的职位名称。"
}

def test_entity_recognition(sentence):
    """测试句子中的实体识别"""
    print(f"\n测试句子: {sentence}")
    
    for entity_type, instruction in entity_types.items():
        # 构建提示
        prompt = f"{instruction}\n输入: {sentence}\n输出:"
        
        # 生成回复
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.1
        )
        
        # 解码回复
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取输出部分
        if "输出:" in response:
            output_part = response.split("输出:")[-1].strip()
            print(f"{entity_type}: {output_part}")

if __name__ == "__main__":
    print("测试Qwen3.5-9B模型的实体识别能力")
    print("=" * 50)
    
    for sentence in test_sentences:
        test_entity_recognition(sentence)
    
    print("\n测试完成！")