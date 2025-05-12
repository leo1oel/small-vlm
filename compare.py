import torch
from transformers import AutoModelForCausalLM

from src.vlm.models.model import VLM

device = "cuda"
dtype = torch.bfloat16
model = VLM.from_pretrained("/pasteur/u/yiming/small-vlm/outputs/2025-05-06/19-26-09").to(
    device, dtype=dtype
)
original_model = AutoModelForCausalLM.from_pretrained(
    "lmsys/vicuna-7b-v1.5",
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    torch_dtype=dtype,
).to(device)

inputs_embeds = torch.load("inputs_embeds.pt").to(device, dtype=dtype)
attention_mask = torch.load("attention_mask.pt").to(device, dtype=dtype)


def detailed_param_comparison(model1, model2, verbose=True, diff_threshold=0):
    """
    比较两个模型的参数差异，并提供详细报告

    参数:
        model1: 第一个模型
        model2: 第二个模型
        verbose: 是否打印每个参数的差异
        diff_threshold: 只报告差异大于此阈值的参数

    返回:
        不同参数的字典，格式为 {param_name: (差异总和, 差异百分比, 参数数量)}
    """
    # 获取两个模型的状态字典
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()

    diff_params = {}
    total_params = 0
    total_diff_params = 0
    overall_diff_sum = 0

    # 检查每个参数的差异
    for key in state_dict1.keys():
        if key not in state_dict2:
            print(f"参数 '{key}' 在模型2中不存在")
            continue

        # 获取参数张量
        param1 = state_dict1[key]
        param2 = state_dict2[key]

        # 检查形状
        if param1.shape != param2.shape:
            print(f"参数 '{key}' 形状不同: {param1.shape} vs {param2.shape}")
            continue

        # 计算差异
        diff = (param1 - param2).abs()
        diff_sum = diff.sum().item()
        num_params = param1.numel()
        total_params += num_params

        # 计算差异统计
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        percent_diff = (diff_sum / (param1.abs().sum().item() + 1e-10)) * 100

        # 如果有差异且超过阈值
        if diff_sum > diff_threshold:
            diff_params[key] = (diff_sum, percent_diff, num_params)
            total_diff_params += 1
            overall_diff_sum += diff_sum

            if verbose:
                print(f"\n参数: {key}")
                print(f"  形状: {param1.shape}, 参数数量: {num_params}")
                print(f"  差异总和: {diff_sum:.6f}")
                print(f"  平均差异: {mean_diff:.6f}")
                print(f"  最大差异: {max_diff:.6f}")
                print(f"  相对差异: {percent_diff:.6f}%")

                # 对于小型参数，可以显示具体的差异值
                if num_params <= 10:
                    print(f"  参数1: {param1.cpu().numpy().tolist()}")
                    print(f"  参数2: {param2.cpu().numpy().tolist()}")
                    print(f"  差异: {diff.cpu().numpy().tolist()}")

    # 总结报告
    print("\n===== 总结报告 =====")
    print(f"总参数数量: {total_params}")
    print(f"不同的参数数量: {total_diff_params}")
    print(f"所有参数的差异总和: {overall_diff_sum:.6f}")
    print(f"参数不同的比例: {(total_diff_params / len(state_dict1)) * 100:.2f}%")

    # 按差异大小排序展示前N个差异最大的参数
    if diff_params:
        print("\n差异最大的前10个参数:")
        sorted_diffs = sorted(diff_params.items(), key=lambda x: x[1][0], reverse=True)
        for i, (key, (diff_sum, percent_diff, num_params)) in enumerate(sorted_diffs[:10]):
            print(
                f"{i + 1}. {key}: 差异总和={diff_sum:.6f}, 相对差异={percent_diff:.6f}%, 参数数量={num_params}"
            )

    return diff_params


if __name__ == "__main__":
    detailed_param_comparison(model._language_model._language_model, original_model)
    vlm_config = model._language_model._language_model.config
    original_config = original_model.config

    if vlm_config.to_dict() == original_config.to_dict():
        print("Model configurations are identical.")
    else:
        print("Model configurations differ!")
        # 你可以使用 difflib 来查看具体差异
        import difflib
        from pprint import pformat

        diff = difflib.unified_diff(
            pformat(vlm_config.to_dict()).splitlines(),
            pformat(original_config.to_dict()).splitlines(),
            fromfile="vlm_config",
            tofile="original_config",
            lineterm="",
        )
        print("\n".join(diff))
    print(original_model(attention_mask=attention_mask, inputs_embeds=inputs_embeds))
    print(
        model._language_model._language_model(
            attention_mask=attention_mask, inputs_embeds=inputs_embeds
        )
    )
