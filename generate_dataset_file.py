import os

def create_dataset_file(root_dir, split):
 
    output_file = f'/work/cvcs2024/SelfService_CheckOut/datasets/Food2k_{split}.txt'
    split_dir = os.path.join(root_dir, split)
    
    with open(output_file, 'w') as f:

        for class_dir in sorted(os.listdir(split_dir)):
            class_path = os.path.join(split_dir, class_dir)
            
            if not os.path.isdir(class_path):
                continue
                
            for img_name in sorted(os.listdir(class_path)):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    relative_path = f'{class_dir}/{img_name}'
                    f.write(f'{relative_path} {class_dir}\n')

def main():
    root_dir = '/work/cvcs2024/SelfService_CheckOut/datasets/Food2k_split_dataset'
    splits = ['train', 'test', 'val']
    
    for split in splits:
        create_dataset_file(root_dir, split)
        print(f'Created {split}.txt')

if __name__ == '__main__':
    main()