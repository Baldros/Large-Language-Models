# Unsloth Fine-Tuning Framework for LLMs

## Apresentação

O [**Unsloth**](https://unsloth.ai/) é uma ferramenta projetada para simplificar e acelerar o processo de **_fine-tuning_** de modelos de linguagem (LLMs). Ele oferece uma interface amigável para configurar treinamentos personalizados, permitindo ajustes finos em dados específicos com eficiência e escalabilidade. Com suporte para diversos formatos de dados e integração direta com _frameworks_ populares de _machine learning_, o **Unsloth** torna o _fine-tuning_ mais acessível para desenvolvedores e equipes que buscam aplicar modelos **LLM** em aplicações específicas, otimizando custos e recursos computacionais.

---

## Descrição
Este repositório contém a implementação de um pipeline completo de **_fine-tuning_** para modelos de linguagem de grande porte (**LLMs**) utilizando a biblioteca **Unsloth**. O projeto explora a eficiência e escalabilidade proporcionadas por **LoRA (Low-Rank Adaptation)**, quantização em 4 bits e outras técnicas modernas para ajustar modelos pré-treinados em datasets específicos, como o Alpaca-cleaned.

### Principais Funcionalidades:
1. **Carregamento de modelos otimizados**:
   - Suporte para modelos pré-treinados disponíveis no Hugging Face, incluindo variantes como **Meta-Llama 3.1**, **Mistral**, entre outros.
   - Implementação de quantização em 4 bits para reduzir o consumo de memória.

2. **Fine-tuning com LoRA**:
   - Ajuste eficiente dos modelos com menor uso de memória e maior escalabilidade.
   - Suporte a **gradient checkpointing** e **rank stabilized LoRA (rLoRA)**.

3. **Treinamento supervisionado**:
   - Uso do dataset Alpaca-cleaned para treinar o modelo em tarefas de instrução.
   - Configurações flexíveis para batch size, gradiente acumulado e otimização de memória.

4. **Inferência acelerada**:
   - Habilitação de inferência otimizada para geração de texto rápida e precisa.
