def extract_full_text(soup):
    """Извлекает чистый текст из элемента <main>, удаляя теги и лишние пробелы."""
    main_content = soup.find('main') or soup  # Если <main> нет, берем весь soup

    # Удаляем ненужные элементы
    for element in main_content(['script', 'style', 'meta', 'noscript', 'link', 'header', 'footer']):
        element.decompose()

    # Получаем весь текст и очищаем его
    raw_text = main_content.get_text(' ', strip=True)
    
    # Удаляем лишние пробелы и спецсимволы
    cleaned_text = ' '.join(raw_text.split())
    cleaned_text = re.sub(r'[\n\t\r]+', ' ', cleaned_text)  # Заменяем переносы на пробелы
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Удаляем множественные пробелы
    
    # Разделяем предложения точкой
    sentences = [s.strip() for s in cleaned_text.split('.') if s.strip()]
    result = '. '.join(sentences) + '.' if sentences else ''
    
    print(f"Результат extract_full_text: {result[:200]}...")  # Выводим первые 200 символов для проверки
    return result