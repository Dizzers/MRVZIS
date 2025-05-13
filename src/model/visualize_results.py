import matplotlib.pyplot as plt
import numpy as np
import os

def plot_training_history(results, save_dir='src/model/plots'):
    """Визуализация графиков обучения для всех моделей"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Создаем графики для каждой метрики
    metrics = ['loss', 'val_loss', 'accuracy', 'val_accuracy']
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Сравнение моделей', fontsize=16)
    
    # Цвета для разных моделей
    colors = ['b', 'g', 'r', 'c', 'm']
    
    for idx, metric in enumerate(metrics):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        for i, (model_name, model_results) in enumerate(results.items()):
            history = model_results['history']
            epochs = range(1, len(history[metric]) + 1)
            ax.plot(epochs, history[metric], 
                   label=model_name, 
                   color=colors[i % len(colors)],
                   linewidth=2)
        
        ax.set_title(f'{metric.replace("_", " ").title()}')
        ax.set_xlabel('Эпохи')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_comparison.png'))
    plt.close()
    
    # Создаем график сравнения финальных метрик
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = list(results.keys())
    final_metrics = {
        'Loss': [results[m]['final_loss'] for m in models],
        'Val Loss': [results[m]['final_val_loss'] for m in models],
        'Accuracy': [results[m]['final_accuracy'] for m in models],
        'Val Accuracy': [results[m]['final_val_accuracy'] for m in models]
    }
    
    x = np.arange(len(models))
    width = 0.2
    
    for i, (metric_name, values) in enumerate(final_metrics.items()):
        ax.bar(x + i*width, values, width, label=metric_name)
    
    ax.set_title('Сравнение финальных метрик')
    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels(models, rotation=45)
    ax.legend()
    ax.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'final_metrics_comparison.png'))
    plt.close()
    
    # Создаем тепловую карту для сравнения моделей
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['final_loss', 'final_val_loss', 'final_accuracy', 'final_val_accuracy']
    data = np.array([[results[m][metric] for metric in metrics] for m in models])
    
    # Нормализуем данные для лучшей визуализации
    data_norm = (data - data.min()) / (data.max() - data.min())
    
    im = ax.imshow(data_norm, cmap='YlOrRd')
    
    # Добавляем подписи
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(['Loss', 'Val Loss', 'Accuracy', 'Val Accuracy'])
    ax.set_yticklabels(models)
    
    # Поворачиваем подписи для лучшей читаемости
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Добавляем значения в ячейки
    for i in range(len(models)):
        for j in range(len(metrics)):
            text = ax.text(j, i, f"{data[i, j]:.4f}",
                          ha="center", va="center", color="black")
    
    ax.set_title("Тепловая карта сравнения моделей")
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, 'heatmap_comparison.png'))
    plt.close()
    
    print(f"Графики сохранены в директории: {save_dir}") 