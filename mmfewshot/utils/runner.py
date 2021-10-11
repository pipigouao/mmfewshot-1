import time

from mmcv.runner import EpochBasedRunner
from mmcv.runner.builder import RUNNERS


@RUNNERS.register_module()
class InfiniteEpochBasedRunner(EpochBasedRunner):
    """Epoch-based Runner with a InfiniteSampler.

    This runner train models epoch by epoch.
    """

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        if not hasattr(self, 'data_loader_iter'):
            self.data_loader_iter = iter(self.data_loader)

        for i in range(len(self.data_loader)):
            data_batch = next(self.data_loader_iter)
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=True, **kwargs)
            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1