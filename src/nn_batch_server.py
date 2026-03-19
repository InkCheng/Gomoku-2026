"""
nn_batch_server.py — 神经网络推理批处理服务线程

仿照 Gamma Connect6 的 NN Server 设计:
- 守护线程循环等待推理请求
- 使用 queue.Queue 收集搜索线程的推理请求
- Dynamic Batching: 等待 batch_timeout 或队列达到 max_batch_size 后一次性推理
- 每个请求附带 concurrent.futures.Future, 推理完成后 set_result() 唤醒搜索线程
"""

import threading
import queue
import time
import numpy as np
import torch
from concurrent.futures import Future


class NNBatchServer(threading.Thread):
    """
    神经网络推理批处理服务。

    搜索线程调用 predict_commit() 提交推理请求，获得一个 Future。
    服务线程在后台收集请求，凑够 batch 或超时后一次批量推理，
    然后通过 Future 将结果分发给各搜索线程。
    """

    def __init__(self, model, tss_classifier=None,
                 max_batch_size=8, batch_timeout=0.001,
                 device=None):
        """
        Args:
            model: torch.jit 加载的策略价值网络
            tss_classifier: TSS 分类器模型 (可选)
            max_batch_size: 最大 batch 大小
            batch_timeout: 等待凑 batch 的超时时间 (秒)
            device: torch.device, 默认自动检测
        """
        super().__init__(daemon=True, name="NNBatchServer")
        self.model = model
        self.tss_classifier = tss_classifier
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # 推理请求队列: 每个元素是 (state_tensor, future, request_type)
        # request_type: "policy_value" 或 "tss"
        self._queue = queue.Queue()
        self._stop_event = threading.Event()

        # 统计
        self._total_batches = 0
        self._total_requests = 0

    def predict_commit(self, state_tensor: np.ndarray) -> Future:
        """
        搜索线程调用此方法提交策略价值推理请求。

        Args:
            state_tensor: numpy 数组 (C, H, W)

        Returns:
            Future, 其 result() 返回 (action_probs, value)
                action_probs: numpy 1D array (flatten)
                value: float
        """
        fut = Future()
        self._queue.put((state_tensor, fut, "policy_value"))
        return fut

    def predict_tss(self, state_tensor: np.ndarray) -> Future:
        """
        搜索线程调用此方法提交 TSS 分类推理请求。

        Args:
            state_tensor: numpy 数组

        Returns:
            Future, 其 result() 返回 tss_prob (float)
        """
        fut = Future()
        self._queue.put((state_tensor, fut, "tss"))
        return fut

    def run(self):
        """守护循环: 收集请求 → 拼 batch → model(batch) → 分发结果"""
        print("[NNBatchServer] 推理批处理服务已启动")
        while not self._stop_event.is_set():
            batch = []
            # 阻塞等第一个请求 (最多等 0.1 秒, 防止 shutdown 卡死)
            try:
                item = self._queue.get(timeout=0.1)
                batch.append(item)
            except queue.Empty:
                continue

            # 尝试在 batch_timeout 内凑更多请求
            deadline = time.perf_counter() + self.batch_timeout
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    item = self._queue.get(timeout=remaining)
                    batch.append(item)
                except queue.Empty:
                    break

            # 按 request_type 分组处理
            pv_batch = [(s, f) for s, f, t in batch if t == "policy_value"]
            tss_batch = [(s, f) for s, f, t in batch if t == "tss"]

            if pv_batch:
                self._process_policy_value_batch(pv_batch)
            if tss_batch:
                self._process_tss_batch(tss_batch)

        print("[NNBatchServer] 推理批处理服务已停止")

    def _process_policy_value_batch(self, batch):
        """批量处理策略价值推理"""
        states, futures = zip(*batch)
        try:
            # 拼接为 (N, C, H, W) tensor
            batch_tensor = torch.from_numpy(
                np.stack(states, axis=0)
            ).float().to(self.device)

            with torch.no_grad():
                log_act_probs, values = self.model(batch_tensor)

            act_probs_batch = np.exp(
                log_act_probs.detach().cpu().numpy()
            )  # (N, board_size)
            values_batch = values.detach().cpu().numpy().flatten()  # (N,)

            self._total_batches += 1
            self._total_requests += len(batch)

            # 分发结果
            for i, fut in enumerate(futures):
                if not fut.cancelled():
                    fut.set_result((act_probs_batch[i], values_batch[i]))

        except Exception as e:
            # 出错时通知所有等待线程
            for fut in futures:
                if not fut.done():
                    fut.set_exception(e)

    def _process_tss_batch(self, batch):
        """批量处理 TSS 分类推理"""
        if self.tss_classifier is None:
            for _, fut in batch:
                if not fut.done():
                    fut.set_result(0.0)
            return

        states, futures = zip(*batch)
        try:
            batch_tensor = torch.from_numpy(
                np.stack(states, axis=0)
            ).float().to(self.device)

            with torch.no_grad():
                probs = torch.sigmoid(
                    self.tss_classifier(batch_tensor)
                ).cpu().numpy().flatten()

            for i, fut in enumerate(futures):
                if not fut.cancelled():
                    fut.set_result(float(probs[i]))

        except Exception as e:
            for fut in futures:
                if not fut.done():
                    fut.set_exception(e)

    def shutdown(self):
        """关闭服务"""
        self._stop_event.set()
        self.join(timeout=3.0)
        print(f"[NNBatchServer] 共处理 {self._total_batches} 个 batch, "
              f"{self._total_requests} 个请求")

    @property
    def stats(self):
        return {
            "total_batches": self._total_batches,
            "total_requests": self._total_requests,
            "avg_batch_size": (self._total_requests / max(self._total_batches, 1))
        }
