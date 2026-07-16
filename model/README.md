# AI 话务信息预测 — 训练/推理复现说明

本包 results.csv 的完整生成链(所有超参与模型选择均来自训练集按小区留出验证,
不读取测试标签、不做输入指纹、不跨测试组聚合统计;每组预测仅使用该组 336 行
历史 + 工参/天气可选输入,可逐条复现):

```bash
export P1_DATA=/path/to/AI数据集   # train_data.csv/test_data.csv/parameter.csv/weather.csv
export P1_CACHE=/path/to/cache
cd model/repro_att314
python build_cache_single.py          # 训练窗口特征缓存(单线程)
python build_raw_cache_single.py
python build_test_cache_single.py
cd ..
# 1) FABLE 基线(6xLightGBM两阶段 + 4x卷积NN + 2xMLP, log域集成+钳位+乘子)
python predict.py "$P1_DATA/train_data.csv" "$P1_DATA/test_data.csv" fable_results.csv
# 2) NN 成员导出(attention314 / conv107 / conv-metric2-314)
cd repro_att314
python export_nn_member.py --arch manual_attn --tag export_att314 --epochs 30 --ch 64 --hid 384 --heads 4 --layers 2 --lr 0.0008 --aug 0.08 --noise 0.04 --seed 314 --save-prefix "$P1_CACHE/export_att314" --data-root "$P1_DATA" --cache-root "$P1_CACHE"
python export_nn_member.py --arch conv --tag export_conv107 --epochs 44 --ch 64 --hid 512 --lr 0.001 --aug 0.08 --noise 0.04 --seed 107 --save-prefix "$P1_CACHE/export_conv107" --data-root "$P1_DATA" --cache-root "$P1_CACHE"
python export_nn_member.py --arch conv --tag export_conv_metric2_s314 --metric-only 2 --epochs 44 --ch 64 --hid 512 --lr 0.001 --aug 0.08 --noise 0.04 --seed 314 --save-prefix "$P1_CACHE/export_conv_metric2_s314" --data-root "$P1_DATA" --cache-root "$P1_CACHE"
cd ..
# 3) 合成最终结果(att314 替换 conv107 端点 + 强度外推 + 指标2专家混合 + 钳位/乘子)
python generate_final.py --base-results fable_results.csv --cache-root "$P1_CACHE" --out results.csv
```

随机数说明(审查要点): LightGBM SEEDS=(7,99,1907,4099,777,2718), deterministic=True;
卷积NN (seed,hid)∈{(7,512),(107,512),(2027,512),(2718,768)}; MLP seed∈{7,107};
增量成员 seed: att=314, conv107=107, metric2=314; 全部 torch.manual_seed 驱动,
CPU 单线程 + use_deterministic_algorithms(True) 可逐比特复现(CUDA 下数值可能有
微小浮点差异, 审查复现建议用 CPU)。generate_final.py 中
ATT_STRENGTH/METRIC2_WEIGHT 为训练集留出验证选定的常数。
环境: Python 3.10+, numpy, lightgbm 4.x, torch 2.x。
