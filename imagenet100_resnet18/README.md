ImageNet dataset is downloaded from https://image-net.org/ by registering. After downloaded the train dataset a .tar file, and  is extracted with: 

```
mkdir train
nohup tar xf ILSVRC2012_img_train.tar --directory train >/dev/null 2>&1 &
```
1000 .tar files are created for each class.



Use bash file to subset specified classes in classes.txt.
