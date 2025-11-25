import cv2
import numpy as np
from PIL import Image
import face_alignment
from utils.extractor import get_transform_mat, FaceType
from utils.merger import paste_foreground_onto_background
from utils.segmentation import get_mask_from_landmarks
from utils.operators import apply_blur, apply_mosaic, apply_diffusion

def anonymize_faces_segmented(
    image,
    face_alignment_model,
    mask_features,
    operator_type,
    face_image_size=512,
    pipe=None,
    generator=None,
    dilate_radius=3,
    blur_sigma=0,
    smooth_edges=True,
    **kwargs
):
    """
    Anonimiza rostos em uma imagem usando máscaras segmentadas e operadores específicos.
    
    Esta função detecta rostos na imagem, segmenta as características faciais especificadas
    e aplica um operador de anonimização (blur, mosaic ou diffusion) apenas nas regiões
    segmentadas.
    
    Args:
        image (PIL.Image or np.ndarray): Imagem de entrada.
        face_alignment_model: Modelo face_alignment inicializado.
        mask_features (list): Lista de características para a máscara.
                             Exemplos: ['eyes', 'mouth'], ['nose', 'lips'], etc.
                             Opções disponíveis: 'eyes', 'left_eye', 'right_eye',
                             'eyebrows', 'nose', 'nose_bridge', 'nose_tip', 'nostrils',
                             'mouth', 'lips', 'teeth', 'cheeks', 'forehead'.
        operator_type (str): Tipo de operador de anonimização.
                            - 'blur': Desfoque gaussiano
                            - 'mosaic': Pixelização/mosaico
                            - 'diffusion': Anonimização baseada em difusão
        face_image_size (int): Tamanho da face alinhada (padrão: 512).
        pipe: Pipeline de difusão (obrigatório para operador 'diffusion').
        generator: Gerador PyTorch para reprodutibilidade.
        dilate_radius (int): Raio de dilatação para expandir a máscara (padrão: 3).
        blur_sigma (float): Desvio padrão para suavização gaussiana das bordas (padrão: 0).
        smooth_edges (bool): Se True, aplica suavização nas bordas da máscara (padrão: True).
        kwargs: Argumentos adicionais para os operadores:
                - kernel_size (tuple): Para blur (padrão: (21, 21))
                - block_size (int): Para mosaic (padrão: 10)
                - num_inference_steps (int): Para diffusion (padrão: 50)
                - guidance_scale (float): Para diffusion (padrão: 4.0)
                - anonymization_degree (float): Para diffusion (padrão: 1.25)
    
    Returns:
        PIL.Image: Imagem anonimizada.
    """
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image

    # Garantir RGB
    if image_np.shape[2] == 4:
        image_np = image_np[:, :, :3]
    
    # Detectar landmarks
    # face_alignment espera RGB
    preds = face_alignment_model.get_landmarks(image_np)
    
    if preds is None:
        return Image.fromarray(image_np)
        
    result_np = image_np.copy()
    
    for landmarks in preds:
        # 1. Obter matriz de transformação
        # Usamos WHOLE_FACE como padrão para cobrir área suficiente
        mat = get_transform_mat(landmarks, face_image_size, FaceType.WHOLE_FACE)
        
        # 2. Extrair face alinhada
        face_aligned = cv2.warpAffine(
            result_np,
            mat,
            (face_image_size, face_image_size),
            cv2.INTER_LANCZOS4,
            borderValue=(255, 255, 255),
        )
        
        # 3. Transformar landmarks para o espaço alinhado
        # landmarks é (68, 2). cv2.transform espera (N, 1, 2) ou (1, N, 2)
        pts = np.array([landmarks], dtype=np.float32) # (1, 68, 2)
        # mat é 2x3. cv2.transform funciona.
        aligned_landmarks = cv2.transform(pts, mat)[0] # (68, 2)
        
        # 4. Gerar máscara com os novos parâmetros de segmentação
        mask = get_mask_from_landmarks(
            aligned_landmarks, 
            (face_image_size, face_image_size), 
            mask_features,
            dilate_radius=dilate_radius,
            blur_sigma=blur_sigma,
            smooth_edges=smooth_edges
        )
        
        # 5. Aplicar operador
        if operator_type == 'blur':
            processed_face = apply_blur(face_aligned, mask, **kwargs)
        elif operator_type == 'mosaic':
            processed_face = apply_mosaic(face_aligned, mask, **kwargs)
        elif operator_type == 'diffusion':
            if pipe is None:
                raise ValueError("Pipe deve ser fornecido para o operador diffusion")
            processed_face = apply_diffusion(face_aligned, mask, pipe, generator, **kwargs)
        else:
            raise ValueError(f"Operador desconhecido: {operator_type}")
            
        # 6. Colar de volta
        # paste_foreground_onto_background espera imagens PIL
        result_pil = Image.fromarray(result_np)
        processed_face_pil = Image.fromarray(processed_face)
        
        result_pil = paste_foreground_onto_background(processed_face_pil, result_pil, mat)
        result_np = np.array(result_pil)
        
    return Image.fromarray(result_np)
