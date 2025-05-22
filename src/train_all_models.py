import os
import tensorflow as tf
from model.compare_models import train_and_evaluate_models, print_comparison_results
from model.visualize_results import plot_training_history

def find_midi_files(directory):
    midi_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.mid', '.midi')):
                midi_files.append(os.path.join(root, file))
    return midi_files

def main():
    # Настройка TensorFlow для оптимизации производительности
    tf.config.optimizer.set_jit(True)  # Включаем XLA JIT компиляцию
    tf.config.optimizer.set_experimental_options({
        "layout_optimizer": True,
        "constant_folding": True,
        "shape_optimization": True,
        "remapping": True,
        "arithmetic_optimization": True,
        "dependency_optimization": True,
        "loop_optimization": True,
        "function_optimization": True,
        "debug_stripper": True,
        "disable_model_pruning": False,
        "scoped_allocator_optimization": True,
        "pin_to_host_optimization": True,
        "implementation_selector": True,
        "auto_mixed_precision": True
    })

    # Настройка потоков для CPU
    tf.config.threading.set_inter_op_parallelism_threads(4)
    tf.config.threading.set_intra_op_parallelism_threads(4)

    # Создаем директории, если они не существуют
    os.makedirs("src/data", exist_ok=True)
    os.makedirs("src/model/model_weights", exist_ok=True)
    os.makedirs("src/model/plots", exist_ok=True)

    # Получаем список MIDI файлов рекурсивно
    midi_files = find_midi_files("src/data")
    if not midi_files:
        raise ValueError("MIDI файлы не найдены в директории src/data/ и её поддиректориях")

    print(f'Найдено файлов для обучения: {len(midi_files)}')
    print(f'Будет использовано первых 15 файлов для обучения')

    # Оптимизированные параметры обучения
    print(f'Устанавливаем количество эпох: 50')
    results = train_and_evaluate_models(
        midi_files=midi_files,
        epochs=50,  
        batch_size=64, 
        max_files=15,
        validation_split=0.15  
    )

    # Вывод результатов сравнения
    print_comparison_results(results)

    # Создание визуализаций
    print("\nСоздание графиков сравнения...")
    plot_training_history(results)

if __name__ == "__main__":
    main() 