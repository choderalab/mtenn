import torch


class Combination(torch.nn.Module):
    def __init__(self):
        """
        Stuff that will be common among all Combination subclasses.
        """
        super(Combination, self).__init__()

        self.gradients = {}
        self.predictions = []

    def forward(self, prediction: torch.Tensor, model: torch.nn.Module):
        """
        Takes a prediction and model, and tracks the pred and the gradient of the pred
        wrt model parameters. This part should be the same for all Combination methdos,
        so we can put it in the base class.

        Parameters
        ----------
        prediction : torch.Tensor
            Model prediction
        model : torch.nn.Module
            The model being trained
        """
        # Track prediction
        self.predictions.append(prediction.detach())

        # Don't do anything with gradients if model is in eval mode
        if not model.training:
            return

        # Get gradients (zero first to get rid of any existing)
        model.zero_grad()
        prediction.backward()
        for n, p in model.named_parameters():
            try:
                self.gradients[n].append(p.grad.detach())
            except KeyError:
                self.gradients[n] = [p.grad.detach()]

    def predict(self):
        raise NotImplementedError(
            "A Combination class must have the `predict` method implemented."
        )


class MeanCombination(Combination):
    """
    Combine a list of predictions by taking the mean.
    """

    def __init__(self):
        super(MeanCombination, self).__init__()

    def predict(self, model: torch.nn.Module):
        """
        Returns the mean of all stored predictions, and appropriately sets the model
        parameter grads for an optimizer step.

        Parameters
        ----------
        model : torch.nn.Module
            The model being trained

        Returns
        -------
        torch.Tensor
            Combined prediction (mean of all stored preds)
        """
        # Return mean of all preds
        all_preds = torch.stack(self.predictions).flatten()
        final_pred = all_preds.mean(axis=None).detach()

        if model.training:
            # Calculate final gradient (derivation details are in README_COMBINATION)
            for n, p in model.named_parameters():
                p.grad = torch.stack(self.gradients[n], axis=-1).mean(axis=-1)

            # Reset internal trackers
            self.gradients = {}
        self.predictions = []

        return final_pred, all_preds


class MaxCombination(Combination):
    """
    Approximate max/min of the predictions using the LogSumExp function for smoothness.
    """

    def __init__(self, neg=True, scale=1000.0):
        """
        Parameters
        ----------
        neg : bool, default=True
            Negate the predictions before calculating the LSE, effectively finding
            the min. Preds are negated again before being returned
        scale : float, default=1000.0
            Fixed positive value to scale predictions by before taking the LSE. This
            tightens the bounds of the LSE approximation
        """
        super(MaxCombination, self).__init__()

        self.neg = -1 * neg
        self.scale = scale

    def __repr__(self):
        return f"MaxCombination(neg={self.neg}, scale={self.scale})"

    def __str__(self):
        return repr(self)

    def predict(self, model: torch.nn.Module):
        """
        Returns the max/min of all stored predictions, and appropriately sets the model
        parameter grads for an optimizer step.

        Parameters
        ----------
        model : torch.nn.Module
            The model being trained

        Returns
        -------
        torch.Tensor
            Combined prediction (LSE max approximation of all stored preds)
        """
        # Calculate once for reuse later
        all_preds = torch.stack(self.predictions).flatten()
        adj_preds = self.neg * self.scale * all_preds.detach()
        Q = torch.logsumexp(adj_preds, dim=0)
        # Calculate the actual prediction
        final_pred = (self.neg * Q / self.scale).detach()

        if model.training:
            # Calculate final gradients for each parameter
            final_grads = {}
            for n, p in self.gradients.items():
                final_grads[n] = (
                    torch.stack(
                        [g * (pred - Q).exp() for g, pred in zip(p, adj_preds)],
                        axis=-1,
                    )
                    .detach()
                    .sum(axis=-1)
                )

            # Set weights gradients
            for n, p in model.named_parameters():
                try:
                    p.grad = final_grads[n]
                except RuntimeError as e:
                    print(n, p.grad.shape, final_grads[n].shape, flush=True)
                    raise e

            # Reset internal trackers
            self.gradients = {}
        self.predictions = []

        return final_pred, all_preds


class BoltzmannCombination(Combination):
    """
    Combine a list of deltaG predictions according to their Boltzmann weight.
    Treat energy in implicit kT units.
    """

    def __init__(self):
        super(BoltzmannCombination, self).__init__()

    def predict(self, model: torch.nn.Module):
        """
        Returns the Boltzmann weighted average of all stored predictions, and
        appropriately sets the model parameter grads for an optimizer step.

        Parameters
        ----------
        model : torch.nn.Module
            The model being trained

        Returns
        -------
        torch.Tensor
            Combined prediction (Boltzmann-weighted average)
        """
        # Save for later so we don't have to keep redoing this
        adj_preds = -torch.stack(self.predictions).flatten().detach()

        # First calculate the normalization factor
        Q = torch.logsumexp(adj_preds, dim=0)

        # Calculate w
        w = (adj_preds - Q).exp()

        # Calculate final pred
        final_pred = torch.dot(w, -adj_preds)

        if model.training:
            # Calculate dQ/d_theta
            dQ = {
                n: -torch.stack(
                    [(p - Q).exp() * g for p, g in zip(adj_preds, grads)], axis=-1
                ).sum(axis=-1)
                for n, grads in self.gradients.items()
            }

            # Calculate dw/d_theta
            dw = {
                n: [(p - Q).exp() * (-g - dQ[n]) for p, g in zip(adj_preds, grads)]
                for n, grads in self.gradients.items()
            }

            # Calculate final grads
            final_grads = {}
            for n, p in self.gradients.items():
                final_grads[n] = (
                    torch.stack(
                        [
                            w_grad * -pred + w_val * grad
                            for w_grad, pred, w_val, grad in zip(dw[n], adj_preds, w, p)
                        ],
                        axis=-1,
                    )
                    .detach()
                    .sum(axis=-1)
                )

            # Set weights gradients
            for n, p in model.named_parameters():
                try:
                    p.grad = final_grads[n]
                except RuntimeError as e:
                    print(n, p.grad.shape, final_grads[n].shape, flush=True)
                    raise e

            # Reset internal trackers
            self.gradients = {}
        self.predictions = []

        return final_pred
