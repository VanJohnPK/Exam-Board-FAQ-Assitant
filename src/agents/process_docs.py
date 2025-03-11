import re

# 定义中文数字列表
CHINESE_NUMBERS = ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十']

# 输入文件列表
input_files = ['o_docs/book0.txt', 
               'o_docs/book1.txt',
               'o_docs/book2.txt',
               'o_docs/book3.txt',
               'o_docs/book4.txt',
]

# 遍历输入文件列表
for input_file in input_files:
    # 生成输出文件路径，将 'o_docs' 替换为 'docs'
    output_file = input_file.replace('o_docs', 'docs')

    # 存储各级标题的列表
    headers = []

    # 打开输入文件读取内容
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"文件未找到: {input_file}")
        continue

    # 处理每一行
    processed_lines = []
    for line in lines:
        # 去除行尾的换行符
        line = line.rstrip()

        # 检查是否为一级标题（如“一、秋考”）
        if line.strip().startswith(tuple(f"{i}、" for i in CHINESE_NUMBERS)):
            try:
                # 当遇到一级标题时，更新 headers 列表为只包含该一级标题
                headers = [line.split('、')[1].strip().split('\t')[0]]
            except IndexError:
                print(f"格式错误，无法解析一级标题：{line}")
        # 检查是否为二级标题（如“（一）报名篇”）
        elif line.strip().startswith(tuple(f"（{i}）" for i in CHINESE_NUMBERS)):
            try:
                if len(headers) > 0:
                    # 如果已有一级标题，将二级标题添加到 headers 列表中
                    headers = headers[:1] + [line.split('）')[1].strip()]
                else:
                    # 如果没有一级标题，先输出提示信息，再将二级标题作为唯一标题
                    print(f"警告：遇到二级标题但没有一级标题，当前行：{line}")
                    headers = [line.split('）')[1].strip()]
            except IndexError:
                print(f"格式错误，无法解析二级标题：{line}")
        # 检查行是否以数字和点开头（支持多位数）
        elif re.match(r'^\d+\.\s', line.strip()):
            # 生成标签
            tags = ''.join([f"【{header}】" for header in headers])
            # 在行尾添加标签
            line = line + tags
            print(line)

        # 将处理后的行添加到列表中
        processed_lines.append(line)

    # 打开输出文件写入处理后的内容
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in processed_lines:
                f.write(line + '\n')
        print(f"处理完成，结果已保存到 {output_file}")
    except Exception as e:
        print(f"写入文件时出错: {output_file}, 错误信息: {e}")