from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型和分词器
print("正在加载模型...")
model = AutoModelForCausalLM.from_pretrained('/home/coke/code/BANER/mods/Qwen3.5-9B', device_map='auto')
tokenizer = AutoTokenizer.from_pretrained('/home/coke/code/BANER/mods/Qwen3.5-9B')
print("模型加载成功！")

# 测试聊天功能
print("\n开始聊天测试，输入exit退出")
while True:
    prompt = input('用户: ')
    if prompt == 'exit':
        break
    # 处理输入并生成回复
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print('模型: ' + response)

print("聊天测试结束")