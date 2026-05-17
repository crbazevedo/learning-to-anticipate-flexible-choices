"""
Experiment Logger Module

Provides comprehensive logging functionality for portfolio optimization experiments
with structured logging, multiple log levels, and performance tracking.
"""

import logging
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import traceback

class ExperimentLogger:
    """Comprehensive logger for portfolio optimization experiments."""
    
    def __init__(self, experiment_id: str, log_dir: str = "experiments/logs"):
        """
        Initialize the experiment logger.
        
        Args:
            experiment_id: Unique identifier for the experiment
            log_dir: Directory to store log files
        """
        self.experiment_id = experiment_id
        self.log_dir = Path(log_dir) / experiment_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize loggers for different components
        self.loggers = {}
        self.setup_loggers()
        
        # Performance tracking
        self.start_time = time.time()
        self.performance_metrics = {}
        
        # Log experiment start
        self.log_experiment_start()
    
    def setup_loggers(self):
        """Setup individual loggers for different components."""
        log_levels = {
            'experiment': logging.INFO,
            'algorithm': logging.DEBUG,
            'portfolio': logging.INFO,
            'performance': logging.INFO,
            'learning': logging.DEBUG,
            'metrics': logging.INFO,
            'data': logging.INFO,
            'error': logging.ERROR
        }
        
        for component, level in log_levels.items():
            logger = logging.getLogger(f"{self.experiment_id}.{component}")
            logger.setLevel(level)
            
            # Create file handler
            log_file = self.log_dir / f"{component}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            
            # Add handler to logger
            logger.addHandler(file_handler)
            
            # Store logger
            self.loggers[component] = logger
    
    def log_experiment_start(self):
        """Log experiment initialization."""
        self.log('experiment', 'INFO', f"Experiment {self.experiment_id} started", {
            'timestamp': datetime.now().isoformat(),
            'experiment_id': self.experiment_id,
            'log_directory': str(self.log_dir)
        })
    
    def log(self, component: str, level: str, message: str, data: Optional[Dict[str, Any]] = None):
        """
        Log a message with structured data.
        
        Args:
            component: Logger component (experiment, algorithm, etc.)
            level: Log level (DEBUG, INFO, WARNING, ERROR)
            message: Log message
            data: Additional structured data to log
        """
        if component not in self.loggers:
            self.loggers[component] = logging.getLogger(f"{self.experiment_id}.{component}")
        
        logger = self.loggers[component]
        
        # Create structured log entry
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'component': component,
            'level': level,
            'message': message,
            'data': data or {}
        }
        
        # Log to file
        if level.upper() == 'DEBUG':
            logger.debug(json.dumps(log_entry))
        elif level.upper() == 'INFO':
            logger.info(json.dumps(log_entry))
        elif level.upper() == 'WARNING':
            logger.warning(json.dumps(log_entry))
        elif level.upper() == 'ERROR':
            logger.error(json.dumps(log_entry))
        else:
            logger.info(json.dumps(log_entry))
    
    def log_algorithm_start(self, algorithm_name: str, params: Dict[str, Any]):
        """Log algorithm execution start."""
        self.log('algorithm', 'INFO', f"Algorithm {algorithm_name} started", {
            'algorithm': algorithm_name,
            'parameters': params,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_algorithm_progress(self, generation: int, metrics: Dict[str, Any]):
        """Log algorithm progress."""
        self.log('algorithm', 'DEBUG', f"Generation {generation} completed", {
            'generation': generation,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_algorithm_end(self, algorithm_name: str, final_metrics: Dict[str, Any], duration: float):
        """Log algorithm execution end."""
        self.log('algorithm', 'INFO', f"Algorithm {algorithm_name} completed", {
            'algorithm': algorithm_name,
            'final_metrics': final_metrics,
            'duration_seconds': duration,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_portfolio_construction(self, portfolio_weights: Dict[str, float], metrics: Dict[str, Any]):
        """Log portfolio construction."""
        self.log('portfolio', 'INFO', "Portfolio constructed", {
            'weights': portfolio_weights,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_portfolio_rebalancing(self, old_weights: Dict[str, float], new_weights: Dict[str, float], 
                                 transaction_costs: float):
        """Log portfolio rebalancing."""
        self.log('portfolio', 'INFO', "Portfolio rebalanced", {
            'old_weights': old_weights,
            'new_weights': new_weights,
            'transaction_costs': transaction_costs,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_performance_metrics(self, metrics: Dict[str, Any], period: str = "daily"):
        """Log performance metrics."""
        self.log('performance', 'INFO', f"Performance metrics calculated ({period})", {
            'metrics': metrics,
            'period': period,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_learning_event(self, event_type: str, data: Dict[str, Any]):
        """Log anticipatory learning events."""
        self.log('learning', 'DEBUG', f"Learning event: {event_type}", {
            'event_type': event_type,
            'data': data,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_metrics_collection(self, metrics_type: str, data: Dict[str, Any]):
        """Log metrics collection."""
        self.log('metrics', 'INFO', f"Metrics collected: {metrics_type}", {
            'metrics_type': metrics_type,
            'data': data,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_data_loading(self, dataset_name: str, data_info: Dict[str, Any]):
        """Log data loading events."""
        self.log('data', 'INFO', f"Data loaded: {dataset_name}", {
            'dataset': dataset_name,
            'info': data_info,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Log errors with full context."""
        error_data = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'context': context or {},
            'timestamp': datetime.now().isoformat()
        }
        
        self.log('error', 'ERROR', f"Error occurred: {str(error)}", error_data)
    
    def log_experiment_end(self, summary: Dict[str, Any]):
        """Log experiment completion."""
        duration = time.time() - self.start_time
        
        self.log('experiment', 'INFO', f"Experiment {self.experiment_id} completed", {
            'duration_seconds': duration,
            'summary': summary,
            'timestamp': datetime.now().isoformat()
        })
        
        # Save experiment summary
        summary_file = self.log_dir / "experiment_summary.json"
        with open(summary_file, 'w') as f:
            json.dump({
                'experiment_id': self.experiment_id,
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_seconds': duration,
                'summary': summary
            }, f, indent=2)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get collected performance metrics."""
        return self.performance_metrics.copy()
    
    def add_performance_metric(self, metric_name: str, value: Any):
        """Add a performance metric."""
        self.performance_metrics[metric_name] = value
    
    def save_performance_metrics(self):
        """Save performance metrics to file."""
        metrics_file = self.log_dir / "performance_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.performance_metrics, f, indent=2)
    
    def get_log_files(self) -> List[str]:
        """Get list of all log files."""
        return [str(f) for f in self.log_dir.glob("*.log")]
    
    def cleanup(self):
        """Clean up logger resources."""
        for logger in self.loggers.values():
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler) 