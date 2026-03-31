import json
import os
from data.loader import DataLoader
from analysis.analyzer import Analyzer, ModelDiagnosis
from suggestions.suggester import Suggester
from core.fix_engine import FixEngine
from tasks.base import BaseTask
from models.base import BaseModel

class IterativeLoop:
    def __init__(self, model: BaseModel, task: BaseTask, data_loader: DataLoader, 
                 max_iterations: int = 5, improvement_threshold: float = 0.01,
                 verbose: bool = True):
        self.model = model
        self.task = task
        self.data_loader = data_loader
        self.max_iterations = max_iterations
        self.improvement_threshold = improvement_threshold
        self.verbose = verbose
        
        self.analyzer = Analyzer()
        self.suggester = Suggester()
        
        self.results = []
        self.best_score = -float('inf')
        self.best_model_name = ""

    def run(self):
        if self.verbose:
            print("\n" + "="*50)
            print("  STARTING ITERATIVE REFINEMENT LOOP (IRL)")
            print("="*50)
            
        X_train, X_test, y_train, y_test = self.data_loader.load_and_split()
        
        for i in range(1, self.max_iterations + 1):
            model_name = type(self.model).__name__
            
            # 1. Train
            self.model.train(X_train, y_train)
            
            # 2. Evaluate
            train_metrics = self.task.evaluate(self.model, X_train, y_train)
            val_metrics = self.task.evaluate(self.model, X_test, y_test)
            
            current_score = val_metrics.get("r2") or val_metrics.get("accuracy") or 0
            
            # 3. Diagnose
            diagnosis = self.analyzer.diagnose(train_metrics, val_metrics)
            
            # 4. Suggest
            suggestion = self.suggester.suggest(diagnosis, model_name)
            
            # Store Result
            iteration_result = {
                "iteration": i,
                "model": model_name,
                "score": round(current_score, 4),
                "diagnosis": diagnosis,
                "action": suggestion['action'],
                "metrics": val_metrics
            }
            self.results.append(iteration_result)

            # Update Best
            if current_score > self.best_score:
                self.best_score = current_score
                self.best_model_name = model_name

            # Output if verbose
            if self.verbose:
                print(f"\n[ITERATION {i}]")
                print(f"  Model:     {model_name}")
                print(f"  Score:     {iteration_result['score']}")
                print(f"  Diagnosis: {diagnosis}")
                print(f"  Action:    {suggestion['action']}")

            # Check Stop Conditions
            if diagnosis == ModelDiagnosis.ACCEPTABLE:
                if self.verbose: print("\n>>> Stopping: Performance is acceptable.")
                break
            
            if i > 1 and current_score <= (self.results[-2]['score'] + self.improvement_threshold):
                if self.verbose: print(f"\n>>> Stopping: No significant improvement (delta < {self.improvement_threshold}).")
                break
                
            # 5. Fix
            self.model = FixEngine.apply_fix(self.model, suggestion)
            
        self._save_results()
        self._print_final_summary()
        return self.model

    def _save_results(self, filename="results.json"):
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=4)
        if self.verbose:
            print(f"\nDetailed results saved to {filename}")

    def _print_final_summary(self):
        print("\n" + "="*50)
        print("  FINAL SUMMARY")
        print("="*50)
        print(f"  Best Model: {self.best_model_name}")
        print(f"  Best Score: {round(self.best_score, 4)}")
        print("="*50 + "\n")
