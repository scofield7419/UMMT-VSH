# Scene Graph as Pivoting: Inference-time Image-free Unsupervised Multimodal Machine Translation with Visual Scene Hallucination

---


### Step 0.  install prerequisites
```
conda env create -f environments/full.yml
conda activate UMMT-VSH
pip install -e fairseq/
pip install -e taming-transformers/ 
```

### Step 1. prepare data

- MMT data
    - Multi30k

- NMT data with image source
    - WMT14 En→De, En→Fr 
    - WMT16 1032 En→Ro
    - WIT-images


### Step 2. preprocess data 

- Binarize translation data for fairseq
  ```sh
  bash scripts/multi30k/preproc.sh
  ```
- Download Flickr30K Flickr30K and MS-COCO image, then create symbolic link
  ```sh
  ln -s /xxx/flickr30k
  ln -s /xxx/mscoco
  ```


- Download WIT translation data from with parallel corpora organized for machine translation. The archive also includes tokenized and BPE encoded sentences.
- For each translation task, download images in `[train|valid|test]_url.txt` to corresponding paths provided in `[train|valid|test]_img.txt`. Image filenames are the MD5 hashes of their URLs.
- Binarize translation data for fairseq
  ```sh
  bash scripts/wit/preproc.sh
  ```
  


### Step 3. SG parsing for data 

parse the SG structures for all images and texts by the tools in `SG-parsing/VSG` and `SG-parsing/LSG`.


### Step 4. train system

- run `scripts/multi30k-train.sh` script for multi30k
- run `scripts/wmt-train.sh` script for wmt



### Step 5. test with system

- run `scripts/test.sh` script