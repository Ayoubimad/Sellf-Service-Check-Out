import torch
from train_siamese import SiameseTrainer
from config import Config
from collections import defaultdict

def analyze_triplets():

    config = Config()
    trainer = SiameseTrainer(config)

    batch_stats = defaultdict(list)

    mining_types = ['semihard']
    
    with torch.no_grad():
        trainer.model.eval()
        
        for mining_type in mining_types:
            print(f"\nAnalyzing {mining_type} mining...")
            
            for batch_idx, (images, targets) in enumerate(trainer.train_dataloader):
                unique_classes = len(torch.unique(targets))
                
                embeddings = trainer.get_embeddings_in_chunks(images)
                triplets = trainer.get_triplets(embeddings, targets, mining_type, 0)
                
                batch_stats['mining_type'].append(mining_type)
                batch_stats['batch_idx'].append(batch_idx)
                batch_stats['batch_size'].append(len(targets))
                batch_stats['unique_classes'].append(unique_classes)
                batch_stats['num_triplets'].append(len(triplets) if len(triplets) > 0 else 0)
                
                print(
                    f"Batch {batch_idx:3d} | "
                    f"Mining: {mining_type:8s} | "
                    f"Batch Size: {len(targets):3d} | "
                    f"Unique Classes: {unique_classes:3d} | "
                    f"Triplets Generated: {len(triplets) if len(triplets) > 0 else 0:5d}"
                )

if __name__ == "__main__":
    analyze_triplets() 
 