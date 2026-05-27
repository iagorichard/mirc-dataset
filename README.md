# MIRC Dataset

You can see the main MIRC dataset webpage for download at Hugging Face: https://huggingface.co/datasets/iagorrs/mirc

### Download and Setup Dataset

1. You can clone this repository to handle MIRC dataset and navigate to it:
> git clone https://github.com/iagorichard/mirc-dataset.git
> 
> cd mirc-dataset

2. Download the images and annotations:
> curl -L -o annotations.zip https://huggingface.co/datasets/iagorrs/mirc/resolve/main/annotations.zip
> 
> curl -L -o videos.zip https://huggingface.co/datasets/iagorrs/mirc/resolve/main/videos.zip

3. Unzip downloaded files:
> unzip annotations.zip -d annotations
>
> unzip videos.zip -d videos

4. Remove zip files after unziped them:
> rm annotations.zip
>
> rm videos.zip
