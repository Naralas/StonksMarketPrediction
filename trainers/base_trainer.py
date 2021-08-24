import wandb

class BaseTrainer:
    """This class is the base inherited trainer class for both Pytorch and Keras models.
    It contains functions to train models, evaluate and make predictions.

    Do not instatiate this, instead implement the train(self), predict(self) and evaluate(self) functions.
    """
    def __init__(self, model, use_wandb=True, project_label="Default_Project_Name", verbose=False):
        """Init function for the trainer.

        Args:
            model (Pytorch or Keras model object, inheriting models.base_model.BaseModel in the project): 
            use_wandb (bool, optional): Flag to use weights and biases for tracking metrics. You will have to create an API key. Defaults to True.
            project_label (str, optional): Project label, used by wandb. Defaults to "Default_Project_Name".
            verbose (bool, optional): Verbosity, will print metrics if true. Defaults to False.
        """
        self.use_wandb=use_wandb
        self.verbose = verbose
        if use_wandb:
            wandb.init(project=f"{project_label}", config=model.config)
            print(f"Wandb run page : {wandb.run.get_url()}")
        self.model = model
    
    def train(self):
        raise NotImplementedError()

    def predict(self):
        raise NotImplementedError()

    def evaluate(self):
        raise NotImplementedError()