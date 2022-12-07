import numpy as np
import torch

from .model import Model

# Model contains model object and model call


class Trainer:
    """
    Trainer object containing loss function, optimizer, and training function.
    """

    def __init__(
        self, loss_func, optimizer, model, device="cpu", early_stopping=None
    ):
        super(Trainer, self).__init__()
        self.loss_func: torch.nn.modules.loss._Loss = loss_func
        self.optimizer: torch.optim.Optimizer = optimizer
        self.model: Model = model

        self.device = torch.device(device)
        self.model.model = self.model.model.to(device)

    def _train_one(
        self, train_datasets, eval_datasets, train_targets, eval_targets
    ):
        """
        One iteration of a training loop. This gives a minimal example, but
        subclasses of Trainer should overload this method to implement
        application-specific features.

        Parameters
        ----------
        train_datasets: Iter[torch.utils.data.Dataset]
            Some iterable containing the datasets to train on. Backprop will be
            run using these datasets.
        eval_datasets: Iter[torch.utils.data.Dataset]
            Some iterable containing the datasets to evaluate on. Backprop will
            NOT be run using these datasets.
        train_targets: Iter[torch.Tensor]
            Iterable containing target prediction tensors (one tensor for each
            dataset in `train_datasets`).
        eval_targets: Iter[torch.Tensor]
            Iterable containing target prediction tensors (one tensor for each
            dataset in `eval_datasets`).

        Returns
        -------
        List[torch.Tensor]
            List of calculated losses for each dataset in `train_datasets`
        List[torch.Tensor]
            List of calculated losses for each dataset in `eval_datasets`
        """

        train_loss = []
        for ds, target_tensor in zip(train_datasets, train_targets):
            tmp_loss = []
            for d, target in zip(ds, target_tensor):
                self.optimizer.zero_grad()
                pred = self.model(d)
                loss = self.loss_func(pred, target)
                tmp_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()
            train_loss.append(torch.tensor(tmp_loss))

        eval_loss = []
        with torch.no_grad():
            for ds, target_tensor in zip(eval_datasets, eval_targets):
                tmp_loss = []
                for d, target in zip(ds, target_tensor):
                    pred = self.model(d)
                    loss = self.loss_func(pred, target)
                    tmp_loss.append(loss.item())
                eval_loss.append(torch.tensor(tmp_loss))

        return (train_loss, eval_loss)

    def train(
        self,
        train_datasets,
        eval_datasets,
        train_targets,
        eval_targets,
        n_epochs,
        save_file=None,
    ):
        """
        Full training process.

        Parameters
        ----------
        train_datasets: Iter[torch.utils.data.Dataset]
            Some iterable containing the datasets to train on. Backprop will be
            run using these datasets.
        eval_datasets: Iter[torch.utils.data.Dataset]
            Some iterable containing the datasets to evaluate on. Backprop will
            NOT be run using these datasets.
        train_targets: Iter[torch.Tensor]
            Iterable containing target prediction tensors (one tensor for each
            dataset in `train_datasets`).
        eval_targets: Iter[torch.Tensor]
            Iterable containing target prediction tensors (one tensor for each
            dataset in `eval_datasets`).
        n_epochs: int
            Number of epochs to train for.
        save_file : str, optional
            Where to save model weights and errors at each epoch. If a directory is
            passed, the weights will be saved as {epoch_idx}.th and the train/test
            losses will be saved as train_err.pkl and test_err.pkl. If a string is
            passed containing {}, it will be formatted with the epoch number.
            Otherwise, the weights will be saved as the passed string

        Returns
        -------
        List[torch.Tensor]
            List of calculated losses for each epoch for each dataset in
            `train_datasets`
        List[torch.Tensor]
            List of calculated losses for each epoch for each dataset in
            `eval_datasets`
        """

        train_loss = None
        eval_loss = None
        for idx in range(n_epochs):
            print(f"Epoch {idx}/{n_epochs}", flush=True)
            if idx % 10 == 0 and idx > 0:
                train_loss_print = " ".join(
                    [f"{t[-1,:].mean():0.5f}" for t in train_loss]
                )
                eval_loss_print = " ".join(
                    [f"{t[-1,:].mean():0.5f}" for t in eval_loss]
                )
                print(f"Train losses: {train_loss_print}")
                print(f"Eval losses: {eval_loss_print}")

            epoch_train_loss, epoch_eval_loss = self._train_one(
                train_datasets, eval_datasets, train_targets, eval_targets
            )
            if train_loss is None:
                train_loss = [
                    t.detach().reshape((1, -1)) for t in epoch_train_loss
                ]
                eval_loss = [
                    t.detach().reshape((1, -1)) for t in epoch_eval_loss
                ]
            else:
                train_loss = [
                    torch.vstack((t1, t2.detach()))
                    for t1, t2 in zip(train_loss, epoch_train_loss)
                ]
                eval_loss = [
                    torch.vstack((t1, t2.detach()))
                    for t1, t2 in zip(eval_loss, epoch_eval_loss)
                ]

            if save_file is None:
                continue
            elif os.path.isdir(save_file):
                torch.save(
                    self.model.model.state_dict(), f"{save_file}/{epoch_idx}.th"
                )
                pkl.dump(train_loss, open(f"{save_file}/train_loss.pkl", "wb"))
                pkl.dump(val_loss, open(f"{save_file}/val_loss.pkl", "wb"))
                pkl.dump(test_loss, open(f"{save_file}/test_loss.pkl", "wb"))
            elif "{}" in save_file:
                torch.save(
                    self.model.model.state_dict(), save_file.format(epoch_idx)
                )
            else:
                torch.save(self.model.model.state_dict(), save_file)

        return (train_loss, eval_loss)
