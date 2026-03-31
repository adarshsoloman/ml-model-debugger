from models.regression import LinearRegressionModel, RandomForestModel, DecisionTreeModel
from models.classification import LogisticRegressionModel, RandomForestClassifierModel, DecisionTreeClassifierModel
from models.base import BaseModel

class FixEngine:
    @staticmethod
    def apply_fix(current_model: BaseModel, suggestion: dict) -> BaseModel:
        """
        Applies improvements: Model switching or Hyperparameter adjustments.
        """
        action = suggestion.get("action")
        
        # Scenario 1: Increase Complexity (Model Switching)
        if action == "Increase model complexity":
            if isinstance(current_model, LinearRegressionModel):
                print(">>> Fix: Switching from Linear to Random Forest.")
                return RandomForestModel(n_estimators=100, random_state=42)
            if isinstance(current_model, DecisionTreeModel):
                print(">>> Fix: Increasing Tree complexity.")
                current_model.set_params({"max_depth": 20})
                return current_model
        
        # Scenario 2: Reduce Complexity (Hyperparameter Tuning)
        elif action == "Apply regularization or reduce complexity":
            print(">>> Fix: Applying regularization/reducing depth.")
            if hasattr(current_model, 'set_params'):
                current_model.set_params({"max_depth": 5})
            return current_model
                
        return current_model
