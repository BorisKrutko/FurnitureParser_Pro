import spacy
import warnings

warnings.filterwarnings('ignore')

def main(parsed_text, nlp): 
    print(type(nlp))
    try:
        if parsed_text:  # Если передан текст от парсера
            try:
                # Обрабатываем текст с помощью модели
                doc = nlp(parsed_text)
                res = []
                
                for ent in doc.ents:
                    if ent.label_ == 'PRODUCT':
                        res.append(ent.text)
            
                return res

            except Exception as e:
                print(f"Ошибка при обработке текста: {e}")

        else:
            print("Пожалуйста, предоставьте текст для анализа (parsed_text).")

    except Exception as e:
        print(f"Произошла общая ошибка: {e}")

def ner_classification(my_parsed_text, nlp, classifier, is_with_classification):
    if is_with_classification:
        products = main(my_parsed_text, nlp) 
        print("PRODUCTS=====================================================================")
        print(products)
        
        categories = ["furniture"]
        results = classifier(products, candidate_labels=categories, multi_label=False)

        clean_products = []
        for product, result in zip(products, results):
            if result['scores'][0] >= 0.8:
                clean_products.append(product)
        print(clean_products)
        
            
        # Классификация
        categories = [
            'sofa', 
            'chair',
            'table',
            'bed'
        ]
        results = classifier(clean_products, candidate_labels=categories, multi_label=False)

        # Создание словаря для результатов
        categorized_products = {category: [] for category in categories}
        categorized_products["all"] = []

        # Распределение товаров по категориям
        for product, result in zip(clean_products, results):
            best_category = result['labels'][0]
            confidence = result['scores'][0]
            
            if confidence > 0.8:
                categorized_products[best_category].append(product)
                categorized_products["all"].append(product)
                
        print(categorized_products)    
        return categorized_products
    else:
        products = main(my_parsed_text, nlp) 
        print("=====================================================================PRODUCTS")
        print(products)
        
        #categories = ["product"]
        #results = classifier(products, candidate_labels=categories, multi_label=False)

        clean_products = {'all': []}
        for product in products: #zip(products, results):
            #if result['scores'][0] >= 0.8:
            clean_products['all'].append(product)
        
        return clean_products