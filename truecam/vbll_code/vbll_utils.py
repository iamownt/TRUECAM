import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import vbll
from torch.utils.data import DataLoader, TensorDataset
from sklearn import metrics
from sklearn.metrics import balanced_accuracy_score


def create_balanced_sampler(y_train):
    all_labels = y_train.numpy()
    class_counts = np.bincount(all_labels)
    weights_per_class = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = weights_per_class[all_labels]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler



class VBLLMLP:
    """Base class for VBLL models with a simple API."""

    def __init__(self,
                 input_dim,
                 hidden_dim=128,
                 output_dim=10,
                 num_layers=2,
                 model_type='discriminative',
                 reg_weight=0.01,
                 parameterization='diagonal',
                 return_ood=True,
                 prior_scale=1.0,
                 device=None):
        """
        Initialize a VBLL model.

        Args:
            input_dim: Number of input features
            hidden_dim: Hidden layer dimensions
            output_dim: Number of output classes
            num_layers: Number of hidden layers
            model_type: 'discriminative' or 'generative'
            reg_weight: Regularization weight
            parameterization: Type of parameterization for VBLL
            return_ood: Whether to return OOD scores
            prior_scale: Scale for the prior
            device: Device to run the model on
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.model_type = model_type
        self.reg_weight = reg_weight
        self.parameterization = parameterization
        self.return_ood = return_ood
        self.prior_scale = prior_scale

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else \
                "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device

        # Create the appropriate model
        self._create_model()

    def _create_model(self):
        """Create the appropriate VBLL model based on the specified type."""
        if self.model_type == 'discriminative':
            self.model = self._create_discriminative_model()
        elif self.model_type == 'generative':
            self.model = self._create_generative_model()
        else:
            raise ValueError(f"Model type '{self.model_type}' not supported yet")

        self.model = self.model.to(self.device)

    def _create_discriminative_model(self):
        """Create discriminative VBLL model."""

        class DiscVBLLMLP(nn.Module):
            def __init__(self, cfg):
                super(DiscVBLLMLP, self).__init__()

                self.params = nn.ModuleDict({
                    'in_layer': nn.Linear(cfg.input_dim, cfg.hidden_dim),
                    'core': nn.ModuleList([nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
                                           for _ in range(cfg.num_layers)]),
                    'out_layer': vbll.DiscClassification(
                        cfg.hidden_dim, cfg.output_dim, cfg.reg_weight,
                        parameterization=cfg.parameterization,
                        return_ood=cfg.return_ood,
                        prior_scale=cfg.prior_scale
                    ),
                })
                self.activations = nn.ModuleList([nn.ELU() for _ in range(cfg.num_layers)])
                self.cfg = cfg

            def forward(self, x):
                if len(x.shape) > 2:
                    x = x.view(x.shape[0], -1)

                x = self.params['in_layer'](x)

                for layer, ac in zip(self.params['core'], self.activations):
                    x = ac(layer(x))

                return self.params['out_layer'](x)

        # Create config object to pass parameters
        cfg = type('Config', (), {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'num_layers': self.num_layers,
            'reg_weight': self.reg_weight,
            'parameterization': self.parameterization,
            'return_ood': self.return_ood,
            'prior_scale': self.prior_scale
        })

        return DiscVBLLMLP(cfg)

    def _create_generative_model(self):
        """Create generative VBLL model."""

        class GenVBLLMLP(nn.Module):
            def __init__(self, cfg):
                super(GenVBLLMLP, self).__init__()

                self.params = nn.ModuleDict({
                    'in_layer': nn.Linear(cfg.input_dim, cfg.hidden_dim),
                    'core': nn.ModuleList(
                        [nn.Linear(cfg.hidden_dim, cfg.hidden_dim) for _ in range(cfg.num_layers)]),
                    'out_layer': vbll.GenClassification(cfg.hidden_dim, cfg.output_dim, cfg.reg_weight,
                                                        parameterization=cfg.parameterization,
                                                        return_ood=cfg.return_ood,
                                                        prior_scale=cfg.prior_scale),
                })
                self.activations = nn.ModuleList([nn.ELU() for _ in range(cfg.num_layers)])
                self.cfg = cfg

            def forward(self, x):
                if len(x.shape) > 2:
                    x = x.view(x.shape[0], -1)
                x = self.params['in_layer'](x)

                for layer, ac in zip(self.params['core'], self.activations):
                    x = ac(layer(x))

                return self.params['out_layer'](x)

        # Create config object to pass parameters
        cfg = type('Config', (), {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'num_layers': self.num_layers,
            'reg_weight': self.reg_weight,
            'parameterization': self.parameterization,
            'return_ood': self.return_ood,
            'prior_scale': self.prior_scale
        })

        return GenVBLLMLP(cfg)

    def fit(self, X_train, y_train, val_data=None,
            batch_size=512, epochs=30, lr=3e-3, weight_decay=1e-4,
            verbose=1, val_freq=1, monitor='val_acc', save_best=True, balance_sampling=False):
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training labels
            val_data: Tuple of (X_val, y_val) or None
            batch_size: Batch size for training
            epochs: Number of training epochs
            lr: Learning rate
            weight_decay: Weight decay regularization
            verbose: Verbosity level (0, 1, 2)
            val_freq: Validation frequency (in epochs)

        Returns:
            History of training metrics
        """
        # Convert inputs to tensors if they're not already
        if not isinstance(X_train, torch.Tensor):
            X_train = torch.tensor(X_train, dtype=torch.float32)
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.tensor(y_train, dtype=torch.long)

        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        if balance_sampling:
            train_sampler = create_balanced_sampler(y_train)
        else:
            train_sampler = None
        shuffle_train_data = train_sampler is None

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train_data,
                                  num_workers=8, pin_memory=True, sampler=train_sampler)

        # Create validation data loader if provided
        if val_data is not None:
            X_val, y_val = val_data
            if not isinstance(X_val, torch.Tensor):
                X_val = torch.tensor(X_val, dtype=torch.float32)
            if not isinstance(y_val, torch.Tensor):
                y_val = torch.tensor(y_val, dtype=torch.long)
            val_dataset = TensorDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        else:
            val_loader = None

        # Set up optimizer
        param_list = [
            {'params': self.model.params.in_layer.parameters(), 'weight_decay': weight_decay},
            {'params': self.model.params.core.parameters(), 'weight_decay': weight_decay},
            {'params': self.model.params.out_layer.parameters(), 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(param_list, lr=lr)
        total_iter = len(train_loader) * epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=total_iter, eta_min=0)
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_bacc': [] if monitor == 'val_bacc' else None
        }

        best_metric_value = float('-inf') if monitor != 'val_loss' else float('inf')
        best_model_state = None
        best_epoch = -1

        for epoch in range(epochs):
            # Training
            self.model.train()
            running_loss = []
            running_acc = []

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                output = self.model(X_batch)

                loss = output.train_loss_fn(y_batch)
                probs = output.predictive.probs
                acc = (torch.argmax(probs, dim=1) == y_batch).float().mean().item()

                running_loss.append(loss.item())
                running_acc.append(acc)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                optimizer.step()
                scheduler.step()

            epoch_train_loss = np.mean(running_loss)
            epoch_train_acc = np.mean(running_acc)
            history['train_loss'].append(epoch_train_loss)
            history['train_acc'].append(epoch_train_acc)

            # Validation
            if val_loader is not None and (epoch % val_freq == 0 or epoch == epochs - 1):
                self.model.eval()
                val_loss = []
                val_acc = []
                all_preds = []
                all_targets = []

                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)

                        output = self.model(X_batch)
                        loss = output.val_loss_fn(y_batch)
                        probs = output.predictive.probs
                        preds = torch.argmax(probs, dim=1)
                        acc = (preds == y_batch).float().mean().item()

                        val_loss.append(loss.item())
                        val_acc.append(acc)

                        # Store predictions and targets for bacc calculation if needed
                        if monitor == 'val_bacc':
                            all_preds.extend(preds.cpu().numpy())
                            all_targets.extend(y_batch.cpu().numpy())

                epoch_val_loss = np.mean(val_loss)
                epoch_val_acc = np.mean(val_acc)
                history['val_loss'].append(epoch_val_loss)
                history['val_acc'].append(epoch_val_acc)

                epoch_val_bacc = None
                if monitor == 'val_bacc':
                    epoch_val_bacc = balanced_accuracy_score(
                        np.array(all_targets),
                        np.array(all_preds)
                    )
                    history['val_bacc'].append(epoch_val_bacc)

                current_metric_value = None
                if monitor == 'val_loss':
                    current_metric_value = epoch_val_loss
                    is_better = current_metric_value < best_metric_value
                elif monitor == 'val_acc':
                    current_metric_value = epoch_val_acc
                    is_better = current_metric_value > best_metric_value
                elif monitor == 'val_bacc':
                    current_metric_value = epoch_val_bacc
                    is_better = current_metric_value > best_metric_value

                if current_metric_value is not None and is_better:
                    best_metric_value = current_metric_value
                    best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    best_epoch = epoch
                    if verbose >= 2:
                        print(f"New best model found at epoch {epoch + 1} with {monitor} = {best_metric_value:.4f}")

                if verbose >= 1:
                    log_msg = f"Epoch {epoch + 1}/{epochs} - loss: {epoch_train_loss:.4f} - acc: {epoch_train_acc:.4f}"
                    log_msg += f" - val_loss: {epoch_val_loss:.4f} - val_acc: {epoch_val_acc:.4f}"
                    if epoch_val_bacc is not None:
                        log_msg += f" - val_bacc: {epoch_val_bacc:.4f}"
                    print(log_msg)
            elif verbose >= 1:
                print(f"Epoch {epoch + 1}/{epochs} - loss: {epoch_train_loss:.4f} - acc: {epoch_train_acc:.4f}")

        # Restore best model weights if requested and we have validation data
        if save_best and best_model_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_model_state.items()})
            if verbose >= 1:
                print(
                    f"Restored best model from epoch {best_epoch + 1} with {monitor} = {best_metric_value:.4f}")

        # Clean up any None values from history
        history = {k: v for k, v in history.items() if v is not None}
        return history

    def predict(self, X, return_probs=False, return_uncertainty=False, batch_size=512):
        """
        Generate predictions from the model.

        Args:
            X: Input features
            return_probs: Whether to return probability distributions
            return_uncertainty: Whether to return uncertainty estimates
            batch_size: Batch size for prediction

        Returns:
            Predictions and optionally probabilities and uncertainty
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)

        dataset = TensorDataset(X)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        preds_list = []
        probs_list = []
        uncertainty_list = []

        self.model.eval()
        with torch.no_grad():
            for (X_batch,) in dataloader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)

                probs = outputs.predictive.probs
                preds = torch.argmax(probs, dim=1)

                preds_list.append(preds.cpu().numpy())

                if return_probs:
                    probs_list.append(probs.cpu().numpy())

                if return_uncertainty:
                    if hasattr(outputs, 'ood_scores'):
                        uncertainty = outputs.ood_scores
                        # print("Using OOD scores for uncertainty estimation.")
                    else:
                        # Fallback uncertainty measure
                        uncertainty = -torch.max(probs, dim=1)[0]
                        print("Using max probability for uncertainty estimation.")
                    uncertainty_list.append(uncertainty.cpu().numpy())

        results = np.concatenate(preds_list)

        if return_probs or return_uncertainty:
            return_values = [results]
            if return_probs:
                return_values.append(np.concatenate(probs_list))
            if return_uncertainty:
                return_values.append(np.concatenate(uncertainty_list))
            return tuple(return_values)

        return results

    def evaluate(self, X, y, ood_data=None, batch_size=512):
        """
        Evaluate the model on test data.

        Args:
            X: Test features
            y: Test labels
            ood_data: Optional out-of-distribution data (X_ood, y_ood)
            batch_size: Batch size for evaluation

        Returns:
            Dictionary of evaluation metrics
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.long)

        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        self.model.eval()
        losses = []
        accs = []

        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                output = self.model(X_batch)
                loss = output.val_loss_fn(y_batch)
                probs = output.predictive.probs
                acc = (torch.argmax(probs, dim=1) == y_batch).float().mean().item()

                losses.append(loss.item())
                accs.append(acc)

        results = {
            'loss': np.mean(losses),
            'accuracy': np.mean(accs)
        }

        # Calculate OOD detection metrics if OOD data is provided
        if ood_data is not None:
            X_ood, y_ood = ood_data
            if not isinstance(X_ood, torch.Tensor):
                X_ood = torch.tensor(X_ood, dtype=torch.float32)

            # Get uncertainty scores for in-distribution data
            _, _, uncertainty_in = self.predict(X, return_uncertainty=True, batch_size=batch_size)

            # Get uncertainty scores for OOD data
            _, _, uncertainty_ood = self.predict(X_ood, return_uncertainty=True, batch_size=batch_size)

            # Compute AUROC for OOD detection
            labels = np.concatenate([np.zeros_like(uncertainty_in), np.ones_like(uncertainty_ood)])
            scores = np.concatenate([uncertainty_in, uncertainty_ood])

            fpr, tpr, _ = metrics.roc_curve(labels, scores)
            auroc = metrics.auc(fpr, tpr)
            results['ood_auroc'] = auroc

        return results

    def save(self, path):
        """Save model to file."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'output_dim': self.output_dim,
                'num_layers': self.num_layers,
                'model_type': self.model_type,
                'reg_weight': self.reg_weight,
                'parameterization': self.parameterization,
                'return_ood': self.return_ood,
                'prior_scale': self.prior_scale
            }
        }, path)

    @classmethod
    def load(cls, path, device=None):
        """Load model from file."""
        checkpoint = torch.load(path, map_location='cpu')
        config = checkpoint['model_config']

        model = cls(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            output_dim=config['output_dim'],
            num_layers=config['num_layers'],
            model_type=config['model_type'],
            reg_weight=config['reg_weight'],
            parameterization=config['parameterization'],
            return_ood=config['return_ood'],
            prior_scale=config['prior_scale'],
            device=device
        )

        model.model.load_state_dict(checkpoint['model_state_dict'])
        return model


# Example usage:
'''
# Create the model
model = VBLLMLP(
    input_dim=train_features.shape[1],  
    output_dim=10,  # Assuming 10 classes
    model_type='discriminative'
)

# Train the model
history = model.fit(
    train_features, 
    train_labels, 
    val_data=(val_features, val_labels),
    epochs=30
)

# Evaluate the model
metrics = model.evaluate(test_features, test_labels)
print(f"Test accuracy: {metrics['accuracy']:.4f}")

# Make predictions
predictions = model.predict(test_features)
# With probabilities and uncertainty
predictions, probabilities, uncertainties = model.predict(
    test_features, 
    return_probs=True, 
    return_uncertainty=True
)

# Save the model
model.save('vbll_model.pt')

# Load the model
loaded_model = VBLLMLP.load('vbll_model.pt')
'''
