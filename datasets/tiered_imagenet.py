import os
import pickle
from collections import OrderedDict
from collections import defaultdict
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

from .oxford_pets import OxfordPets
import numpy as np

@DATASET_REGISTRY.register()
class TieredImageNet(DatasetBase):

    dataset_dir = "tiered-imagenet/tiered_imagenet"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir)
        self.preprocessed = os.path.join(self.dataset_dir, "preprocessed.pkl")
        self.label_to_classnames = os.path.join(self.dataset_dir, "label_to_classnames.pkl") # add label to classname mapping
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.preprocessed):
            with open(self.preprocessed, "rb") as f:
                preprocessed = pickle.load(f)
                train = preprocessed["train"]
                val = preprocessed["val"]
                test = preprocessed["test"]
                print("load from preprocessed")
        else:
            text_file = os.path.join(self.dataset_dir, "classnames.txt")
            classnames = self.read_classnames(text_file)
            train = self.read_data(classnames, "train")
            # Follow standard practice to perform evaluation on the val set
            # In tiered imagenet we have splitted class
            val = self.read_data(classnames, "val")
            test = self.read_data(classnames, "test")
            print("zhuoyan=== tiered_imagenet dataset: ", len(classnames)) # [1000]

            preprocessed = {"train": train, "val": val, "test": test}
            with open(self.preprocessed, "wb") as f:
                pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)
            print("save preprocessed")

        ### label to classnames mapping, not used right now
        '''
        if os.path.exists(self.label_to_classnames):
            with open(self.label_to_classnames, "rb") as f:
                self.label_to_classnames_dict = pickle.load(f)
                print("load from label_to_classnames")
        else:
            text_file = os.path.join(self.dataset_dir, "classnames.txt")
            classnames = self.read_classnames(text_file)

            label_to_classnames = {}
            for split in ['train', 'val', 'test']:
                label_to_classnames[split] = defaultdict(list) ####=============== not done
                split_dir = os.path.join(self.image_dir, split)
                folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
                for label, folder in enumerate(folders):
                    classname = classnames[folder]
                    label_to_classnames[split][label] = classname
            print(label_to_classnames) ## zhuoyan added, will delete later
            self.label_to_classnames_dict = label_to_classnames
            with open(self.label_to_classnames, "wb") as f:
                pickle.dump(label_to_classnames, f, protocol=pickle.HIGHEST_PROTOCOL)
            print("save label_to_classnames")
        '''
                

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train = data["train"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                data = {"train": train}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        
        ### sampling part
        # Now there are 124,261 imgs in val and 206,209 in test, too many, sample 50 images per class
        print("start sampling part dataset")
        sampled_idx = {}
        ##### cache label file since it's time consuming
        for split_tag, num_class in zip(["val", "test"],[97,160]):
            cache_label_file = os.path.join(self.dataset_dir,"cached_{}_labels_vl-tiered-imagenet.npy".format(split_tag))
            if os.path.exists(cache_label_file):
                self.label = np.load(cache_label_file)
                print(
                    f"Loading labels from cached file {cache_label_file}"
                )
            else:
                print(
                    "cannot find cached label file !!!", cache_label_file
                )

            self.catlocs = tuple()
            for cat in range(num_class):
                self.catlocs += (np.argwhere(self.label == cat).reshape(-1),)
            cats = np.arange(num_class)
            ids = []
            for c in cats:
                ids += np.random.choice(self.catlocs[c], 50, replace=False).tolist()
            
            sampled_idx[split_tag] = ids
        
        # print(sampled_idx['val'][50:70])
        val = np.array(val, dtype=object)[sampled_idx['val']].tolist()
        test = np.array(test, dtype=object)[sampled_idx['test']].tolist()
        
        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        if subsample == "base":
            super().__init__(train_x=train, val=train, test=train) # only use train and val split in base train
        elif subsample == "new":
            super().__init__(train_x=test, val=test, test=test) # only use test split
        else:
            raise NotImplementedError("Not found class characteristics!")

    @staticmethod
    def read_classnames(text_file):
        """Return a dictionary containing
        key-value pairs of <folder name>: <class name>.
        """
        classnames = OrderedDict()
        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                folder = line[0]
                classname = " ".join(line[1:])
                classnames[folder] = classname
        return classnames

    def read_data(self, classnames, split_dir):
        split_dir = os.path.join(self.image_dir, split_dir)
        folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
        items = []

        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(split_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(split_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items
    
