<h1 align="center">
  <h1 align="center">Detector de Emoções pela Webcam</h1>
  <br>
</h1>

## 🧠 **Sobre o Projeto**
O Detector de Emoções Faciais é uma aplicação de inteligência artificial desenvolvida para reconhecer expressões faciais humanas em tempo real por meio da webcam. Utilizando técnicas de visão computacional e redes neurais convolucionais (CNNs), o sistema é capaz de identificar emoções como alegria, tristeza, raiva, surpresa, medo, desgosto e neutro.

🔍 Funcionalidades principais:
📸 Captura de vídeo em tempo real com a webcam;

😐 Detecção de rostos utilizando Haar Cascade (OpenCV);

🧠 Classificação de emoções com um modelo CNN treinado no dataset FER2013;

📊 Exibição da emoção identificada com a porcentagem de confiança ao lado do nome;

🟥 Quando a confiança for baixa, o rosto é marcado com retângulo vermelho indicando incerteza;

⚙️ Implementação 100% local em Python com arquivos .py, sem depender de nuvem.

🚀 Aplicações e utilidade:

-Estudos sobre inteligência artificial e emoções humanas;

-Projetos educacionais com foco em machine learning e deep learning;

-Sistemas interativos de análise emocional em tempo real;

-Protótipos de interfaces responsivas para feedback emocional.

## ⚙️ **Clonar repositório**

Rode o comando abaixo:

```bash
git clone https://github.com/enzopcz2/chatbot-furia.git
cd chatbot-furia
```

## ⚙️ **Instalar requerimentos**

Instale as dependências do projeto usando o pip:

```bash
pip install -r requirements.txt
```

## ⚙️ **Treino modelo**

OBSERVAÇÃO: Esse passo pode ser pulado, porque o modelo já foi treinado e está presente no arquivo model_emocoes.h5. Contudo, caso queira treinar a IA, apague o arquivo model_emocoes.h5 e siga o tutorial abaixo:

### Passo 1: Baixar dataset

Baixe o dataset presente no link: https://www.kaggle.com/datasets/msambare/fer2013. Coloque a pasta dataset no mesmo diretório que train_model.py

```bash
git clone https://github.com/enzopcz2/chatbot-furia.git
cd chatbot-furia
```

### Passo 2: Treinar o modelo

NOTA: Sinta-se livre para modificar o número de épocas e o tamanho do lote do modelo como preferir.

Rode o seguinte comando:

```bash
python train_model.py
```

ou

```bash
python3 train_model.py
```

E pronto, o modelo começara o treinamento. Isso pode levar algum tempo.

## ⚙️ **Programa principal**

Com o modelo treinado, e model_emocoes.h5 gerado. Rode o comando abaixo:

```bash
python detector_emocao.py
```

ou

```bash
python3 detector_emocao.py
```

E pronto, sua webcam aparecerá e em cima dela, a em evidência estará a emoção que foi identificada na imagem.

## ⚠️ **Importante**
Dependendo de como é sua webcam, o seguinte trecho do código pode causar problemas:

```bash
cap = cv2.VideoCapture(0)
```

Caso sua webcam não funcione, tente trocar o 0 por -1, 1 ou 2. Isso deve fazer funcionar perfeitamente.
