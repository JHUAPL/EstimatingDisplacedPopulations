# Tools for IDP and Refugee Camp Analysis

This code was developed by [JHU/APL](http://www.jhuapl.edu) to demonstrate how artificial intelligence and machine learning can be leveraged for IDP and Refugee Camp planning and safety monitoring.

## References

If you use our code, please cite our [paper](https://arxiv.org/abs/2006.14547):
```
@article{hadzic2020estimating,
  title={Estimating Displaced Populations from Overhead},
  author={Hadzic, Armin and Christie, Gordon and Freeman, Jeffrey and Dismer, Amber and Bullard, Stevan and Greiner, Ashley and Jacobs, Nathan and Mukherjee, Ryan},
  journal={arXiv preprint arXiv:2006.14547},
  year={2020}
}
```

## Dataset

For details and instructions for downloading the dataset used in this work, please see [dataset](dataset/).

## Dependencies

Retrieve submodule files using:
```
git submodule update --init --recursive
```

See [requirements.txt](requirements.txt) and install using:
```
pip install -r requirements.txt
```

## Running the code

Start by modifying the path information in `params.py` to ensure that it is valid, pointing to the location where you have downloaded and formatted the IOM data.

Run the code using:
```
python main.py
```
* `--train` to train the model
* `--test` to evaluate the model performance
* `--augmentation` to perform data augmentation during training
* `--multiprocessing` leverage multiple processes to speed up dataloading
* `--add-osm` use OSM structure mask data
* `--gpu [GPU_ID(s)]` specify GPU device IDs to use

## License

The license is Apache 2.0. See [LICENSE](LICENSE).

