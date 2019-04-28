import numpy as np

class Normalize:
    def fit(self, X):
        self.min = np.min(X)
        self.max = np.max(X)
        
        return self
    
    def transform(self, X):
        X_norm = np.copy(X)
        n_cols = X.shape[1]
        
        for i in range(n_cols):
            X_norm[:, i] = (X[:, i] - self.min) / self.max - self.min
        
        return X_norm
    
class Standardize:
    def fit(self, X):
        self.mean = np.mean(X)
        self.std = np.std(X)
        
        return self
    
    def transform(self, X):
        X_std = np.copy(X)
        n_cols = X.shape[1]
        
        for i in range(n_cols):
            X_std[:, i] = (X[:, i] - self.mean) / self.std
            
        return X_std