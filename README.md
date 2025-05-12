# ctreating the dataset

## first step (creating aditional parset to find product names in own catalog)

-   find all links in home page (that have list of subcatalog and all
    products)

``` python
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup

def get_all_links(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # check errors
        
        soup = BeautifulSoup(response.text, 'html.parser')
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            absolute_url = urljoin(url, href)
            
            # Фильтруем якорные ссылки (например, #section)
            if not absolute_url.startswith('#'):
                links.append(absolute_url)

        return list(set(links)) 
    
    except requests.RequestException as e:
        print(f"Ошибка при загрузке страницы: {e}")
        return []


if __name__ == "__main__":
    url = "https://mollanbros.co.uk" 
    all_links = get_all_links(url)
    
    # save to file
    with open("links.txt", "w") as f:
        f.write("\n".join(all_links))
    print("\nВсе ссылки сохранены в файл links.txt")
```

::: {.output .stream .stdout}

    Все ссылки сохранены в файл links.txt
:::
:::

::: {#d1ba63c1 .cell .markdown}
after the parsing delete some links, that doesn\`t relate with this
catalog

    examples
    https://www.nopcommerce.com/
    https://mollanbros.co.uk/register?returnUrl=%2F
:::

::: {#983236e4 .cell .markdown}
### remark

In many catalogs, tags containing product names share the same **class
name**. This provides an opportunity to efficiently locate all products
by searching for these links.

![image.png](vertopal_37bfc42aa4934aa2b3580cc3b78b5997/image.png)

### ⚠ Important Note: {#-important-note}

If you want to parse a different catalog, you must either:

-   Update the class name in the code, or
-   Modify the search structure to match the new catalog's HTML layout.
:::

::: {#11877292 .cell .code execution_count="4"}
``` python
import pandas as pd
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup


def is_valid_url(url):
    """Проверяет, является ли URL валидным."""
    parsed = urlparse(url)
    return bool(parsed.scheme) and bool(parsed.netloc)

# variant of anither aditional parser
""" 
def extract_product_titles(soup):
    product_titles = []
    for link in soup.find_all('p', class_="product-item__title"):
        title = ' '.join(link.get_text(strip=True).split())
        product_titles.append(title)
    return ', '.join(product_titles) if product_titles else ''
"""

def extract_product_titles(soup):
    product_titles = []

    # in h2
    for sp in soup.find_all('h2', class_="product-title"):
        title = ' '.join(sp.get_text(strip=True).split())
        product_titles.append(title)
    
    return ', '.join(product_titles) if product_titles else ''

def parser(url):
    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        return {'product_titles': extract_product_titles(soup)}
    
    except Exception as e:
        print(f"Ошибка при обработке {url}: {e}")
        return {'product_titles': ''}

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.google.com/',
    'Accept-Encoding': 'identity'  
}

if __name__ == "__main__":
    data = pd.read_csv('links.csv') 
    results = []

    for count, url in enumerate(data['URL'], 1):
        if is_valid_url(url):
            # print(f"Обработка URL {count}: {url}")
            parsed_data = parser(url)
            if parsed_data == '': continue
            results.append({
                'URL': url, 'Informative_Text': parsed_data['product_titles']
            })

    # save to csv
    result_df = pd.DataFrame(results)
    result_df.dropna()
    result_df.to_csv("extracted_data.csv", index=False)
    print("Данные успешно сохранены в extracted_data.csv")
```

::: {.output .stream .stdout}
    Ошибка при обработке https://www.nopcommerce.com/: 403 Client Error: Forbidden for url: https://www.nopcommerce.com/
    Данные успешно сохранены в extracted_data.csv
:::
:::

::: {#b943a6dd .cell .markdown}
## Second Step (Create clean_parser that reduces all HTML page content but keeps targets)

### Why Clean the Data?

-   Because NER (Named Entity Recognition) models cannot learn
    effectively from large text chunks:
-   Memory Difficulties → Processing long texts requires excessive
    RAM/GPU memory
-   Poor Results → Large contexts dilute the model\'s ability to
    recognize entities accurately

Key Requirements:

-   Many tags containing **product names** have class names containing
    \"\...product\...\"
-   Links to the product\'s own page and their child elements should be
    considered as **product names**
-   **Product names** are written with capital letters (at the
    beginning)
-   **Product names** do not consist of a single word and do not exceed
    15 words
-   **Product names** do not contain these special characters:
    `{'$', '?', '!', '.', '€', '%', '*', '+', '=', '\\'}`
:::

::: {#52802554 .cell .code}
``` python
from bs4 import NavigableString, Tag
banned_symbols = {'$', '?', '!', '.', '€', '%', '*', '+', '=', '\\'}


def get_text_limited_depth(element, max_depth=2):
    texts = []

    def recurse(node, depth):
        if depth > max_depth:
            return
        if isinstance(node, NavigableString):
            text = str(node).strip()
            if text:
                texts.append(text)
        elif isinstance(node, Tag):
            for child in node.children:
                recurse(child, depth + 1)

    recurse(element, depth=0)
    return ' '.join(texts) if texts else ''


def check_and_add(tags_set, text): 
    if not text or not any(c.isalpha() for c in text):
        return
    
    words = text.split()
    if (2 <= len(words) <= 15 and
        not any(sym in text for sym in banned_symbols) and
        text[0].isupper()):  
        clean_text = ' '.join(words) # norm
        tags_set.add(clean_text)


def extract_informative_text(soup):
    main_content = soup.find('main') or soup
    tags = set()
    
    # delete unusfule tags
    for element in main_content(['script', 'style', 'meta', 'noscript', 'link']):
        element.decompose()

    # 1. tags with PRODUCT
    product_classes = ['product', 'item', 'goods', 'card'] 
    for pattern in product_classes:
        for element in main_content.select(f"[class*={pattern}]"):
            text = get_text_limited_depth(element, max_depth=3)
            check_and_add(tags, text)

    # 2. Links <a>
    for link in main_content.find_all('a', href=True):    
        text = get_text_limited_depth(link, max_depth=2)
        check_and_add(tags, text)

    # 3. <h1 h2 h3 h4 .. h6>
    for header in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'strong', 'b']):
        text = get_text_limited_depth(header, max_depth=2)
        check_and_add(tags, text)
    
    return '. '.join(tags) if tags else ""
```
:::

::: {#ded8483e .cell .markdown}
## Third Step (Merge and Create Dataset)
:::

::: {#5f4c855b .cell .code execution_count="6"}
``` python
import pandas as pd


data = pd.read_csv('dataset_result.csv')
data.info()

print("\nПервые 10 строк:")
print(data.head(10))
```

::: {.output .stream .stdout}
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 605 entries, 0 to 604
    Data columns (total 3 columns):
     #   Column            Non-Null Count  Dtype 
    ---  ------            --------------  ----- 
     0   URL               605 non-null    object
     1   Informative_Text  605 non-null    object
     2   Texts             605 non-null    object
    dtypes: object(3)
    memory usage: 14.3+ KB

    Первые 10 строк:
                                                     URL  \
    0  http://www.muuduufurniture.com/products/frame-...   
    1            https://24estyle.com/products/eliza-fan   
    2  https://allwoodfurn.com/products/group-119-rus...   
    3    https://americanbackyard.com/products/gift-card   
    4       https://asplundstore.se/products/fish-bricka   
    5   https://barnabylane.com.au/products/spensley-tan   
    6  https://big-sale-furniture.com/products/amster...   
    7  https://candb.ca/products/10-gel-memory-foam-m...   
    8  https://classicwithatwist.com.au/products/1900...   
    9  https://craftassociatesfurniture.com/products/...   

                                        Informative_Text  \
    0  3Pcs Dark Chocolate and Cream Faux Leather Sof...   
    1  Demo Cabang Pragmatic Play Gacor Akun Demo Oly...   
    2                                         Group #119   
    3                                          Gift Card   
    4                      Fish bricka vit, Fish brickor   
    5  Spensley Dining Chair - Tan, Spensley Dining C...   
    6  Amsterdam Bench 150 x 35 cm BE-150-35-TA ( Cho...   
    7                                 The Upton Mattress   
    8              1900 BENCH, 1900 ARMCHAIR, 1900 CHAIR   
    9               Canadian Modern Lounge Chairs - 1519   

                                                   Texts  
    0  3Pcs Dark Gray Leather Air Fabric Sofa Set. Ai...  
    1  Live Chat (24 Jam). Fashion & Aksesoris Anak. ...  
    2  What's New. Your Cart is Empty. Request a quot...  
    3  Share on Pinterest. Share on Facebook. SaleLay...  
    4  FelKvantiteten måste vara 1 eller mer. Säljare...  
    5  Smith Full Leather - Tan. Kent Bench - Rose Bl...  
    6  TweetTweet on Twitter. Previous slide. ShareSh...  
    7                  Canadian Made. The Upton Mattress  
    8  Size:SizeL:106 W:53 H:90L:106 W:53 H:90L:106 W...  
    9  Demo SubMenu 2. Milo Baughman Vintage Catalogu...  
:::
# NER SPACY Training

## First Step (MarkUp our datasets)

-   special `.spacy` file, in which you have markup tags in all text

```{=html}
<!-- -->
```
    {
        "text": "Купить iPhone 15 Pro Max по выгодной цене",
        "entities": [(6, 20, "PRODUCT")]  # iPhone 15 Pro Max
    }
:::

::: {#4eb0fe57 .cell .code}
``` python
import re
import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
from spacy.util import filter_spans


def mark_and_save_spacy(df, output_path, label="PRODUCT"):
    nlp = spacy.blank("en")
    doc_bin = DocBin()
    total_targets = 0
    found_targets = 0
    skipped_texts = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="📌 Разметка"):
        targets = str(row['Informative_Text'])
        text = str(row['Texts'])
        entities = []

        for target in targets.split(','): # prepare tags
            target = target.strip()
            
            if not target:
                continue
            
            total_targets += 1
            
            matched = False
            for match in re.finditer(re.escape(target), text, flags=re.IGNORECASE): # find matches
                start, end = match.span()                                           
                entities.append((start, end, label))                                
                matched = True
            
            if matched:
                found_targets += 1

        # skip texts without markup targets
        if not entities:
            skipped_texts += 1
            continue

        doc = nlp.make_doc(text)
        spans = [doc.char_span(start, end, label=label) for start, end, label in entities]
        spans = [s for s in spans if s is not None]
        spans = filter_spans(spans)
        doc.ents = spans
        doc_bin.add(doc)

    doc_bin.to_disk(output_path)
    print(f"\n Размеченные данные сохранены в {output_path}")
    print(f"⏭ Пропущено текстов без сущностей: {skipped_texts}")
    if total_targets > 0:
        coverage = found_targets / total_targets * 100
        print(f"Найдено {found_targets}/{total_targets} таргетов ({coverage:.2f}%)")
    else:
        print("Нет ни одного таргета в данных.")
```
:::

::: {#5f5c3d7c .cell .markdown}
## Second Step (Calculate F1 Metric)

![image.png](vertopal_dc9c7547ef9343bfaae9aeac3a5d05f9/image.png)

Precision = TruePositives + FalsePositives / TruePositives

Recall = TruePositives + FalseNegatives / TruePositives ​

-   **True Positives (TP)**: Correctly predicted positive instances.
-   **False Positives (FP)**: Incorrectly predicted positive instances.
-   **True Negatives (TN)**: Correctly predicted negative instances.
-   **False Negatives (FN)**: Incorrectly predicted negative instances.

F_1 = (2 × Precision + Recall) / Precision × Recall
:::

::: {#ce88f0c9 .cell .code execution_count="6"}
``` python
from sklearn.metrics import f1_score

def evaluate_ner_model(nlp, examples, label="PRODUCT", verbose=True):
    y_true = []
    y_pred = []

    for example in examples:
        pred_doc = nlp(example.text)
        pred_ents = {(ent.start_char, ent.end_char) for ent in pred_doc.ents if ent.label_ == label}
        true_ents = {(ent.start_char, ent.end_char) for ent in example.reference.ents if ent.label_ == label}

        y_true.append(true_ents)
        y_pred.append(pred_ents)

    """
    Бинаризация по сущностям (совпадает ли хотя бы одна сущность по позиции)
    Пример:
        Пусть y_true содержит: [{(0, 5)}, {(10, 15)}] (две сущности).
        Пусть y_pred содержит: [{(0, 5)}] (одна сущность).
    
    В результате:
        Для первого примера: true_labels получит 1, pred_labels получит 1.
        Для второго примера: true_labels получит 1, pred_labels получит 0 (потому что y_pred пустой).
    """
    true_labels = []
    pred_labels = []

    for t_set, p_set in zip(y_true, y_pred):
        true_labels.append(1 if t_set else 0)
        pred_labels.append(1 if p_set else 0)

    f1 = f1_score(true_labels, pred_labels)
    if verbose:
        print(f"F1 для {label}: {f1:.2f}")
    return f1
```
:::

::: {#dc31043e .cell .markdown}
### additional features
:::

::: {#54ed1c0c .cell .code execution_count="4"}
``` python
def load_spacy_data(path):
    doc_bin = DocBin().from_disk(path)
    return list(doc_bin.get_docs(spacy.blank("en").vocab))
```
:::

::: {#0cb508d0 .cell .code execution_count="5"}
``` python
import os
#from google.colab import files


# Сохранение и скачивание модели
def save_and_download_model(model, model_name):
    output_dir = f"/content/{model_name}"
    model.to_disk(output_dir)

    # Создаём zip-архив для скачивания
    os.system(f"zip -r {model_name}.zip {output_dir}")
    #files.download(f"{model_name}.zip")
    print(f"Модель {model_name} сохранена и готова к скачиванию.")
```
:::

::: {#ed7074d6 .cell .markdown}
## Third Step (Train PreTrained model)

![image.png](vertopal_dc9c7547ef9343bfaae9aeac3a5d05f9/image.png)

`model_name="en_core_web_sm"` - pretrained model
:::

::: {#660e58dc .cell .code execution_count="7"}
``` python
from spacy.training import Example
from sklearn.model_selection import KFold
from tqdm import trange
import random
import numpy as np


def train_with_cv(input_paths, n_iter=10, model_name="en_core_web_sm", n_folds=2, label="PRODUCT"):
    # load folds
    docs = load_spacy_data(input_paths[0])
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    if not docs:
        print("Нет документов.")
        return None

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(docs), 1):
        print(f"\n Начало обучения для фолда {fold}/{n_folds}")
        train_docs = [docs[i] for i in train_idx]
        test_docs = [docs[i] for i in test_idx]

        nlp = spacy.load(model_name) # pretrained model

        # add pipe ner
        if "ner" not in nlp.pipe_names: 
            ner = nlp.add_pipe("ner", last=True)
        else:
            ner = nlp.get_pipe("ner")

        ner.add_label(label) 

        # model trained by Examples
        train_examples = [Example(doc, doc) for doc in train_docs] 
        test_examples = [Example(doc, doc) for doc in test_docs]

        optimizer = nlp.resume_training() 
        optimizer.L2 = 0.01

        best_f1 = 0
        for epoch in trange(n_iter, desc=f"Обучение фолда {fold}"):
            random.shuffle(train_examples)
            losses = {}
            nlp.update(train_examples, drop=0.3, losses=losses, sgd=optimizer)
            print(f"📉 Фолд {fold} | Epoch {epoch+1} — Loss: {losses.get('ner', 0):.4f}")

            current_f1 = evaluate_ner_model(nlp, test_examples, label, verbose=False)
            if current_f1 > best_f1:
                best_f1 = current_f1

        fold_results.append(best_f1)
        print(f"🏆 Лучший F1 для фолда {fold}: {best_f1:.4f}")

    print("\nРезультаты кросс-валидации:")
    for fold, f1 in enumerate(fold_results, 1):
        print(f"Фолд {fold}: F1 = {f1:.4f}")
    print(f"Средний F1: {np.mean(fold_results):.4f}")

    return nlp
```
:::

::: {#6c279dff .cell .markdown}
## MAIN
:::

::: {#31ebbd5c .cell .code}
``` python
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from spacy.training import Example

def main():
    CSV_PATH = "dataset_result.csv"
    OUTPUT_DIR = "fold_data"
    TEST_SIZE = 0.2
    N_FOLDS = 2
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True) 

    # train/test
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=42)

    # save test .spacy
    test_path = os.path.join(OUTPUT_DIR, "test.spacy")
    mark_and_save_spacy(test_df, test_path)

    # save train .spacy
    train_path = os.path.join(OUTPUT_DIR, "train.spacy")
    mark_and_save_spacy(train_df, train_path)

    trained_nlp = train_with_cv([train_path], 10, N_FOLDS, 42)

    if trained_nlp:
        # save&download model
        trained_nlp.to_disk("trained_model")
        save_and_download_model(trained_nlp, 'trained_model')
        print("Модель сохранена в 'trained_model'")

        # test
        print("\n Тестирование модели:")
        test_docs = load_spacy_data(test_path)
        test_examples = [Example(doc, doc) for doc in test_docs if len(doc.ents) > 0]

        if test_examples:
            test_f1 = evaluate_ner_model(trained_nlp, test_examples, verbose=True)
            print(f"\nРезультаты на тестовой выборке:")
            print(f"F1-score: {test_f1:.4f}")
        else:
            print("!!! Нет документов для тестирования")

if __name__ == "__main__":
    main()
```
:::

::: {#8d108f50 .cell .markdown}
![image.png](vertopal_dc9c7547ef9343bfaae9aeac3a5d05f9/image.png)
