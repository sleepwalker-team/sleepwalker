import torch

from sleepwalker.models.preprocessors.Preprocessor import Preprocessor

class RobustScaler(Preprocessor):
    def __init__(self, lower_quantile: float = 0.25, upper_quantile: float = 0.75, **kwargs):
        super().__init__()
        self.is_initialized = False
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def requires_warmup(self) -> bool:
        return True

    def push(self, new_data: torch.Tensor):
        """
        Update percentiles for each feature using the P² algorithm.
        Assumes new_data has shape (batch_size, num_samples, num_features).
        """
        # Flatten the first two dimensions (batch_size * num_samples, num_features)
        batch_size, num_samples, num_features = new_data.shape
        device = new_data.device

        if not self.is_initialized:
            self.num_features = num_features
            self.n = torch.zeros(num_features)  # To track counts for each feature
            
            # Initialize tensors to store 5 markers (for the P² algorithm) for each feature
            self.marker_heights = torch.zeros(num_features, 5,device=device)
            self.marker_positions = torch.arange(5).expand(num_features, 5).float().to(device)
            self.desired_positions = torch.zeros(num_features, 5,device=device)
            
            self.initialized = torch.zeros(num_features, dtype=torch.bool,device=device)
            self.is_initialized = True
            self.cnt = 0

        new_data_flat = new_data.view(-1, num_features)  # Shape: (batch_size * num_samples, num_features)

        for feature in range(self.num_features):
            feature_data = new_data_flat[:, feature]
            if not self.initialized[feature]:
                # Initialize marker heights with the sorted feature data
                sorted_data = torch.sort(feature_data)[0]
                num_samples_flat = sorted_data.size(0)
                self.marker_heights[feature] = sorted_data[torch.tensor([
                    0, 
                    int(num_samples_flat * self.lower_quantile), 
                    num_samples_flat // 2, 
                    int(num_samples_flat * self.upper_quantile), 
                    -1
                ])]
                self.initialized[feature] = True

            else:
                # Incrementally update the marker positions and heights using the P² algorithm
                self._update_markers(feature, feature_data)

    def _update_markers(self, feature: int, feature_data: torch.Tensor):
        """
        Vectorized incremental update of the 5 markers using the P² algorithm for the specified feature,
        ensuring that marker heights are adjusted based on data influence.
        """
        # Sort feature_data to handle all values at once
        sorted_data, _ = torch.sort(feature_data)

        # Determine insertion points (k) for each value in sorted_data
        k_values = torch.sum(sorted_data.view(-1, 1) >= self.marker_heights[feature], dim=1).clamp(0, 4)

        # Update markers in a vectorized manner for all values in the batch
        self.marker_heights[feature, 0] = torch.min(self.marker_heights[feature, 0], sorted_data[0])
        self.marker_heights[feature, 4] = torch.max(self.marker_heights[feature, 4], sorted_data[-1])

        for i in range(1, 4):
            # Adjust internal marker positions based on the batch influence
            mask = (k_values == i)
            if mask.any():
                # Vectorized adjustment for marker positions
                self._adjust_marker_positions(feature, i, sorted_data[mask])

    def _adjust_marker_positions(self, feature: int, k: int,  values: torch.Tensor):
        """
        Adjust marker heights in a vectorized manner based on the batch influence.
        """
        delta_pos = (self.marker_positions[feature, k + 1] - self.marker_positions[feature, k - 1])
        if delta_pos == 0:
            return  # Avoid division by zero
        
        # Perform parabolic adjustment of marker heights for the batch
        parabolic_adjustment = (values - self.marker_heights[feature, k]) * \
                            (self.marker_heights[feature, k + 1] - self.marker_heights[feature, k - 1]) / delta_pos
        self.marker_heights[feature, k] += parabolic_adjustment.mean()  # Use mean adjustment for the batch
        self.marker_positions[feature, k] += values.size(0)  # Shift marker position by the batch size

    def update(self, data: torch.Tensor):
        self.push(data)
        self.cnt += 1

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """
        Scale the data for each feature based on the running median and IQR.
        Expects input data to have shape (batch_size, num_samples, num_features).
        """
        if self.is_initialized: #and self.cnt > 100
            batch_size, num_samples, num_features = data.shape
            
            # Flatten the first two dimensions for scaling
            data_flat = data.view(-1, num_features)  # Shape: (batch_size * num_samples, num_features)

            # Compute IQR for each feature
            iqr = self.marker_heights[:, 3] - self.marker_heights[:, 1]
            iqr[iqr == 0] = 1  # To avoid division by zero, treat zero IQR as 1

            # Center by the median (marker 2) and scale by IQR
            scaled_data_flat = (data_flat - self.marker_heights[:, 2]) / iqr

            # Reshape scaled data back to the original shape (batch_size, num_samples, num_features)
            scaled_data = scaled_data_flat.view(batch_size, num_samples, num_features)

            return scaled_data
        else: 
            return data
    