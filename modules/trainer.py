import datetime
import os
from abc import abstractmethod
from tqdm import tqdm  
import time
import torch
import pandas as pd
from numpy import inf
import GPUtil
import logging
import subprocess
from modules.dataloaders import R2DataLoader
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetComputeRunningProcesses
logger = logging.getLogger(__name__)
if os.path.exists("logs") == False:
    os.makedirs("logs")
current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
logger.setLevel(logging.DEBUG)
log_file = "logs/"+current_time+'.log'
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.WARNING)

formatter = logging.Formatter('%(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
stream_handler.setFormatter(formatter)

logger.addHandler(stream_handler)

class BaseTrainer(object):
    def create_train_set(self,batch_size):
        self.args.batch_size = batch_size
        return R2DataLoader(self.args, self.tokenizer, split='train', shuffle=True)
    def fit_train_load(self): # 将batch_size以及使用GPU的数量设置为最佳，如果不能保证最低需求，此处会循环。
        retry_interval = self.args.retry_interval 
        # 将所有的GPU按照可用内存进行降序排序

        
        # 尝试加载尽量大的batch，确定dataloader
        success = False
        while(not success):
            success = False
            gpu_memory_avaliable = self.get_current_process_gpu_memory() # 当前进程已经使用的GPU
            gpus = GPUtil.getGPUs()
            for index, gpu in enumerate(gpus):
                gpu_memory_avaliable[index] = gpu_memory_avaliable[index]+gpu.memoryFree
                # 根据GPU的可用内存进行排序
                sorted_gpu_ids = sorted(gpu_memory_avaliable, key=gpu_memory_avaliable.get, reverse=True)
                self.device = "cuda:"+str(sorted_gpu_ids[0])
            logger.info(f"Main GPU:{self.device}")
            if isinstance(self.model, torch.nn.DataParallel):
                self.model = self.model.module
            
            self.model.to(self.device)
            batch_size = self.args.min_batch_size
            try:
                while(batch_size <= self.args.max_batch_size):
                    self.train_dataloader = self.create_train_set(batch_size)
                    images_id, images, reports_ids, reports_masks = next(iter(self.train_dataloader))
                    images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), reports_masks.to(
                    self.device)
                    output = self.model(images=images, targets=reports_ids, mode='train')
                    loss = self.criterion(output, reports_ids, reports_masks)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    success = True
                    batch_size = batch_size * 2
            except Exception as e: # 说明最后一次出错
                batch_size = batch_size // 2
                
                self.train_dataloader = self.create_train_set(batch_size)
                if success == False:
                    logger.error(f"Error in with min_batch_size {batch_size*2}{e},sleep for {retry_interval} seconds")
                    batch_size = self.args.min_batch_size
                    time.sleep(retry_interval)
        logger.info(f'batch:{batch_size}')
        
        # 尝试使用尽量多的显卡（小于ngpus设定）
        self.avaliable_gpus = [sorted_gpu_ids[0]]
        try:
            while(len(self.avaliable_gpus)<self.args.n_gpu):
                self.avaliable_gpus = self.avaliable_gpus + [sorted_gpu_ids[len(self.avaliable_gpus)]]
                self.model = torch.nn.DataParallel(self.model, device_ids=self.avaliable_gpus)
                images_id, images, reports_ids, reports_masks = next(iter(self.train_dataloader))
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), reports_masks.to(
                self.device)
                output = self.model(images=images, targets=reports_ids, mode='train')
                loss = self.criterion(output, reports_ids, reports_masks)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        except Exception as e:
            self.avaliable_gpus.pop()
            if len(self.avaliable_gpus) == 0:
                self.avaliable_gpus = [sorted_gpu_ids[0]]
                self.model = self.model.to(self.device)
            else:
                self.model = torch.nn.DataParallel(self.model, device_ids=self.avaliable_gpus)
        logger.info(f"Avaliable GPUs:{self.avaliable_gpus}")
    def get_current_process_gpu_memory(self):
        nvmlInit()
        # 当前进程 ID
        current_pid = os.getpid()
        # 获取 GPU 设备句柄 (假设使用 GPU 0)
        usage = dict()
        for i in range(4):
            handle = nvmlDeviceGetHandleByIndex(i)
            usage[i] = 0
        # 获取运行在该设备上的所有进程
            processes = nvmlDeviceGetComputeRunningProcesses(handle)
        # 查找当前进程的 GPU 内存使用情况
            for proc in processes:
                if proc.pid == current_pid:
                    mem_used = proc.usedGpuMemory / 1024 / 1024  # 转换为 MB
                    usage[i] = mem_used
        return usage  # 如果当前进程没有占用 GPU，返回 0

    def __init__(self, model, criterion, metric_ftns, optimizer, args,tokenizer):
        self.tokenizer = tokenizer
        self.args = args
        self.model = model
        # # setup GPU device if available, move model into configured device
        # self.device, self.avaliable_gpus = self._prepare_device(args.n_gpu)
        # self.origin_model = model.to(self.device)
        # if len(self.avaliable_gpus) > 1:
        #     self.model = torch.nn.DataParallel(self.origin_model, device_ids=self.avaliable_gpus)
        # 假设 self.model 是 DataParallel 包装的模型

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if args.resume is not None:
            self._resume_checkpoint(args.resume)

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        retry_interval = self.args.retry_interval
        # max_retry_interval = self.args.max_retry_interval
        not_improved_count = 0
        
        epoch = self.start_epoch
        while(epoch <= self.epochs):
            result = None
            try:
                self.fit_train_load()
                result = self._train_epoch(epoch)
            except Exception as e:
                logger.error(f"Error in epoch {epoch}")
                self._save_checkpoint(epoch, save_best=False)
                torch.cuda.empty_cache()
                continue
                
            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)
            self._record_best(log)
            # print logged informations to the screen
            for key, value in log.items():
                print('\t{:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    print("Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                        self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    print("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                        self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)
            epoch += 1
        self._print_best()
        self._print_best_to_file()

    def _print_best_to_file(self):
        crt_time = time.asctime(time.localtime(time.time()))
        self.best_recorder['val']['time'] = crt_time
        self.best_recorder['test']['time'] = crt_time
        self.best_recorder['val']['seed'] = self.args.seed
        self.best_recorder['test']['seed'] = self.args.seed
        self.best_recorder['val']['best_model_from'] = 'val'
        self.best_recorder['test']['best_model_from'] = 'test'

        if not os.path.exists(self.args.record_dir):
            os.makedirs(self.args.record_dir)
        record_path = os.path.join(self.args.record_dir, self.args.dataset_name+'.csv')
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)
        record_table = record_table.append(self.best_recorder['val'], ignore_index=True)
        record_table = record_table.append(self.best_recorder['test'], ignore_index=True)
        record_table.to_csv(record_path, index=False)
        
        
    # def _prepare_device(self, n_gpu_use):
    #     gpus = GPUtil.getGPUs()
    #     min_memory = self.args.min_gpu_memory
    #     list_ids = []
    #     device = "cpu"
    #     for index, gpu in enumerate(gpus):
    #         logger.info(f"GPU {index}: {gpu.memoryFree}MB")
    #         if gpu.memoryFree > min_memory:
    #             list_ids.append(index)
    #             if len(list_ids) == n_gpu_use:
    #                 break
    #     if len(list_ids) > 0:
    #         device = torch.device('cuda:{}'.format(list_ids[0]))
    #     return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict':self.model.module.state_dict() if isinstance(self.model, torch.nn.DataParallel) else self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        
        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            print("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        self.model = self.origin_model
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _record_best(self, log):
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)

        improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][
            self.mnt_metric_test]) or \
                        (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][
                            self.mnt_metric_test])
        if improved_test:
            self.best_recorder['test'].update(log)

    def _print_best(self):
        print('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['val'].items():
            print('\t{:15s}: {}'.format(str(key), value))

        print('Best results (w.r.t {}) in test set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['test'].items():
            print('\t{:15s}: {}'.format(str(key), value))


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader,tokenizer):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args,tokenizer)
        self.args = args
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.tokenizer = tokenizer

    def _train_epoch(self, epoch):

        train_loss = 0
        self.model.train()
        for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(tqdm(self.train_dataloader, desc="Training")):
            images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), reports_masks.to(
                self.device)
            output = self.model(images=images, targets=reports_ids, mode='train')
            a = 1/(batch_idx-5)
            loss = self.criterion(output, reports_ids, reports_masks)
            train_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
        log = {'train_loss': train_loss / len(self.train_dataloader)}

        self.model.eval()
        with torch.no_grad():
            val_gts, val_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(tqdm(self.val_dataloader, desc="Evaling")):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                output = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                val_res.extend(reports)
                val_gts.extend(ground_truths)
            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            log.update(**{'val_' + k: v for k, v in val_met.items()})

        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(tqdm(self.test_dataloader,desc="Testing")):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                output = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})

        self.lr_scheduler.step()

        return log
