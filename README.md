# CPM_LM_bert4keras
在bert4keras下加载CPM_LM模型

## 相关链接

- 清源CPM主页：https://cpm.baai.ac.cn/
- CPM-LM项目：https://github.com/TsinghuaAI/CPM-Generate
- CPM-LM-TF2：https://github.com/qhduan/CPM-LM-TF2
- 科学空间介绍：https://kexue.fm/archives/7912

## 示例代码

完整脚本见：[basic_language_model_cpm_lm.py](https://github.com/bojone/bert4keras/blob/master/examples/basic_language_model_cpm_lm.py)

```python
# 模型路径
config_path = '/root/kg/bert/CPM_LM_2.6B_TF/config.json'
checkpoint_path = '/root/kg/bert/CPM_LM_2.6B_TF/model.ckpt'
spm_path = '/root/kg/bert/CPM_LM_2.6B_TF/chinese_vocab.model'


def pre_tokenize(text):
    """分词前处理函数
    """
    return [
        w.replace(' ', u'\u2582').replace('\n', u'\u2583')
        for w in jieba.cut(text, cut_all=False)
    ]


tokenizer = SpTokenizer(
    spm_path,
    token_start=None,
    token_end=None,
    pre_tokenize=pre_tokenize,
    token_translate={u'\u2583': '<cls>'}
)  # 建立分词器

model = build_transformer_model(
    config_path=config_path, checkpoint_path=checkpoint_path, model='gpt2'
)  # 建立模型，加载权重
```

## 环境依赖

- bert4keras >= 0.9.3
- tensorflow 1.14+ / tensorflow 2.x 均可
- keras / tf.keras 均可
- 推荐组合：tensorflow 1.14 + keras 2.3.1 + bert4keras 0.9.3

## 交流联系

QQ交流群：808623966，微信群请加机器人微信号spaces_ac_cn
