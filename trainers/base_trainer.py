import wandb

class BaseTrainer(object):
    def __init__(self, model, use_wandb=True, project_label="Default_Project_Name", verbose=False):
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