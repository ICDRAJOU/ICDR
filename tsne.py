from text_net.DGRN import TextProjectionHead

import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import clip
import json

# 학습된 모델 ckpt 불러오기
ckpt_path = "ckpt/epoch_289_random100_moreconcat_batch7.pth"  # 학습된 모델의 체크포인트 경로
checkpoint = torch.load(ckpt_path)

# print(checkpoint.keys())

clip_model, preprocess = clip.load("ViT-B/32", device='cuda')
# 동적으로 텍스트 임베딩 차원 가져오기
text_embed_dim = clip_model.text_projection.shape[1]

# Projection Head와 관련된 가중치 불러오기
projection_head = TextProjectionHead(text_embed_dim, 64).cuda()
# 특정 경로에서 가중치 가져오기
projection_head_state_dict = {
    'proj.0.weight': checkpoint['R.body.4.body.4.dgm2.sft.text_proj_head.proj.0.weight'],
    'proj.0.bias': checkpoint['R.body.4.body.4.dgm2.sft.text_proj_head.proj.0.bias'],
    'proj.2.weight': checkpoint['R.body.4.body.4.dgm2.sft.text_proj_head.proj.2.weight'],
    'proj.2.bias': checkpoint['R.body.4.body.4.dgm2.sft.text_proj_head.proj.2.bias'],
}

# JSON 파일에서 불러오기
with open('prompts.json', 'r') as json_file:
    prompts = json.load(json_file)

# 불러온 데이터에서 각 프롬프트 리스트 가져오기
haze_text = prompts["haze_text"]
rain_text = prompts["rain_text"]
low_text = prompts["low_text"]
low_rain_text = prompts["low_rain_text"]
low_haze_text = prompts["low_haze_text"]
haze_rain_text = prompts["haze_rain_text"]
low_haze_rain_text = prompts["low_haze_rain_text"]

# Projection Head에 해당 가중치 로드
projection_head.load_state_dict(projection_head_state_dict)

# t-SNE 시각화를 위한 함수 정의
def plot_tsne(original_embeddings, projected_embeddings, labels):
    # t-SNE 적용: 2D로 변환
    tsne = TSNE(n_components=2, random_state=42, perplexity=2)
    original_tsne = tsne.fit_transform(original_embeddings)
    projected_tsne = tsne.fit_transform(projected_embeddings)
    class_names = ['Haze', 'Rain', 'LowLight', 'LowLight + Haze', 'LowLight + Rain', 'Haze + Rain', 'LowLight + Haze + Rain']

    # 원본 임베딩 시각화 (Before Projection)
    plt.figure(figsize=(8, 5))
    scatter1 = plt.scatter(original_tsne[:, 0], original_tsne[:, 1], c=labels, cmap='tab10')
    plt.title("Before Projection")
    plt.xticks([])  # xticks 제거
    plt.yticks([])  # yticks 제거

    # 클래스에 맞는 레전드 추가 (우측 바깥에)
    handles1, _ = scatter1.legend_elements()
    plt.legend(handles1, class_names, title="Classes", loc='center left', bbox_to_anchor=(1, 0.5))

    # 이미지 파일 저장
    plt.savefig('t-sne_before_projection.png', bbox_inches='tight')
    plt.close()

    # Projection Head 거친 후 임베딩 시각화 (After Projection)
    plt.figure(figsize=(8, 5))
    scatter2 = plt.scatter(projected_tsne[:, 0], projected_tsne[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.title("After Projection")
    plt.xticks([])  # xticks 제거
    plt.yticks([])  # yticks 제거

    # 클래스에 맞는 레전드 추가 (우측 바깥에)
    handles2, _ = scatter2.legend_elements()
    plt.legend(handles2, class_names, title="Classes", loc='center left', bbox_to_anchor=(1, 0.5))

    # 이미지 파일 저장
    plt.savefig('t-sne_after_projection.png', bbox_inches='tight')
    plt.close()

# 모든 텍스트 프롬프트 리스트 생성
text_prompts = haze_text + rain_text + low_text + low_haze_text + low_rain_text + haze_rain_text + low_haze_rain_text

# CLIP 모델로부터 텍스트 임베딩 추출    
text_tokens = clip.tokenize(text_prompts).cuda()
with torch.no_grad():
    original_embeddings = clip_model.encode_text(text_tokens)  # 원본 텍스트 임베딩

# 학습된 Projection Head를 거친 후 임베딩
projected_embeddings = projection_head(original_embeddings).detach().cpu().numpy()

# 원본 임베딩은 그대로 사용 (detach() 추가)
original_embeddings = original_embeddings.detach().cpu().numpy()

# 레이블 정의 (0: haze, 1: rain, 2: low)
# text_len = 21
labels = [0] * len(haze_text) + [1] * len(rain_text) + [2] * len(low_text) \
         + [3] * len(low_haze_text) + [4] * len(low_rain_text) + [5] * len(haze_rain_text) + [6] * len(low_haze_rain_text)  # 각 클래스에 대해 10개의 레이블 부여

# t-SNE 시각화 수행
plot_tsne(original_embeddings, projected_embeddings, labels)