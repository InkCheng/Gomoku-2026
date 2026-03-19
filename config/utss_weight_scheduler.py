class UTSSWeightScheduler:
    def __init__(self, milestones=None, weights=None):
        # 默认的 milestone 和对应的权重（可以自定义传入）
        if milestones is None:
            self.milestones = [2000, 4000, 8000, 12000]  # 自博弈轮次
        else:
            self.milestones = milestones

        if weights is None:
            self.weights = [0.2, 0.3, 0.5, 0.6, 1.0]
        else:
            self.weights = weights

        assert len(self.weights) == len(self.milestones) + 1, "weights 应该比 milestones 多一个"

    def get_weight(self, step):
        for i, milestone in enumerate(self.milestones):
            if step < milestone:
                return self.weights[i]
        return self.weights[-1]


# 示例用法
if __name__ == "__main__":
    scheduler = UTSSWeightScheduler()
    for i in range(0, 15000, 1000):
        print(f"Step: {i:5d}, UTSS Weight: {scheduler.get_weight(i):.2f}")
