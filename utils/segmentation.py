import cv2
import numpy as np

# Mapeamento dos landmarks faciais (68 pontos)
# Baseado no formato padrão do face_alignment
LANDMARK_INDICES = {
    # Contorno do rosto (jawline)
    'jaw': list(range(0, 17)),
    # Sobrancelha direita
    'right_eyebrow': list(range(17, 22)),
    # Sobrancelha esquerda
    'left_eyebrow': list(range(22, 27)),
    # Ponte do nariz
    'nose_bridge': list(range(27, 31)),
    # Ponta do nariz
    'nose_tip': list(range(31, 36)),
    # Olho direito
    'right_eye': list(range(36, 42)),
    # Olho esquerdo
    'left_eye': list(range(42, 48)),
    # Contorno externo da boca
    'mouth_outer': list(range(48, 60)),
    # Contorno interno da boca (lábios)
    'mouth_inner': list(range(60, 68)),
}

def get_mask_from_landmarks(
    landmarks, 
    image_size, 
    features, 
    dilate_radius=3, 
    blur_sigma=0,
    smooth_edges=True
):
    """
    Gera uma máscara binária para as características faciais especificadas.
    
    Esta função usa os 68 landmarks faciais padrão para criar máscaras segmentadas
    que podem ser usadas para anonimização seletiva de diferentes partes do rosto.
    
    Args:
        landmarks (np.ndarray): Array de 68 landmarks faciais com shape (68, 2).
        image_size (tuple): Tamanho da imagem (width, height).
        features (list): Lista de características para incluir na máscara.
                         Opções disponíveis:
                         - 'eyes': Ambos os olhos (esquerdo e direito)
                         - 'left_eye': Apenas olho esquerdo
                         - 'right_eye': Apenas olho direito
                         - 'eyebrows': Ambas as sobrancelhas
                         - 'left_eyebrow': Apenas sobrancelha esquerda
                         - 'right_eyebrow': Apenas sobrancelha direita
                         - 'nose': Nariz completo (ponte + ponta)
                         - 'nose_bridge': Apenas ponte do nariz
                         - 'nose_tip': Apenas ponta do nariz (inclui narinas)
                         - 'nostrils': Apenas narinas (parte inferior do nariz)
                         - 'mouth': Boca completa (contorno externo)
                         - 'lips': Apenas lábios (contorno externo - contorno interno)
                         - 'teeth': Dentes (contorno interno da boca)
                         - 'cheeks': Bochechas (região entre olhos e boca)
                         - 'forehead': Testa (região acima das sobrancelhas)
        dilate_radius (int): Raio de dilatação para expandir a máscara (padrão: 3).
                             Use 0 para desabilitar dilatação.
        blur_sigma (float): Desvio padrão para suavização gaussiana das bordas (padrão: 0).
                            Use 0 para desabilitar suavização.
        smooth_edges (bool): Se True, aplica suavização nas bordas da máscara (padrão: True).
    
    Returns:
        np.ndarray: Máscara binária (0 ou 255) com shape (height, width).
    """
    mask = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
    
    if not isinstance(landmarks, np.ndarray):
        landmarks = np.array(landmarks)
    
    if landmarks.shape[0] != 68:
        raise ValueError(f"Esperado 68 landmarks, recebido {landmarks.shape[0]}")
        
    # Helper para desenhar convex hull de uma região
    def draw_feature(indices, expand=False):
        if len(indices) == 0:
            return
        points = landmarks[indices].astype(np.int32)
        if len(points) < 3:
            # Se há menos de 3 pontos, desenha círculos pequenos
            for pt in points:
                cv2.circle(mask, tuple(pt), 5, 255, -1)
        else:
            hull = cv2.convexHull(points)
            cv2.fillConvexPoly(mask, hull, 255)
            if expand:
                # Expansão adicional para cobrir melhor a região
                kernel = np.ones((5, 5), np.uint8)
                mask_expanded = cv2.dilate(mask, kernel, iterations=1)
                mask[:] = mask_expanded

    # Olhos
    if 'eyes' in features or 'left_eye' in features:
        draw_feature(LANDMARK_INDICES['left_eye'], expand=True)
        
    if 'eyes' in features or 'right_eye' in features:
        draw_feature(LANDMARK_INDICES['right_eye'], expand=True)
    
    # Sobrancelhas
    if 'eyebrows' in features or 'left_eyebrow' in features:
        draw_feature(LANDMARK_INDICES['left_eyebrow'])
        
    if 'eyebrows' in features or 'right_eyebrow' in features:
        draw_feature(LANDMARK_INDICES['right_eyebrow'])
    
    # Nariz
    if 'nose' in features:
        # Nariz completo: ponte + ponta
        draw_feature(LANDMARK_INDICES['nose_bridge'])
        draw_feature(LANDMARK_INDICES['nose_tip'])
    elif 'nose_bridge' in features:
        draw_feature(LANDMARK_INDICES['nose_bridge'])
    elif 'nose_tip' in features:
        draw_feature(LANDMARK_INDICES['nose_tip'])
    
    if 'nostrils' in features:
        # Narinas: apenas a parte inferior do nariz
        draw_feature(LANDMARK_INDICES['nose_tip'])
    
    # Boca
    if 'mouth' in features:
        # Boca completa (contorno externo)
        draw_feature(LANDMARK_INDICES['mouth_outer'])
    
    if 'lips' in features:
        # Lábios: contorno externo menos contorno interno
        lip_mask = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
        pts_outer = landmarks[LANDMARK_INDICES['mouth_outer']].astype(np.int32)
        cv2.fillConvexPoly(lip_mask, cv2.convexHull(pts_outer), 255)
        
        pts_inner = landmarks[LANDMARK_INDICES['mouth_inner']].astype(np.int32)
        if len(pts_inner) >= 3:
            cv2.fillConvexPoly(lip_mask, cv2.convexHull(pts_inner), 0)
        
        mask = cv2.bitwise_or(mask, lip_mask)

    if 'teeth' in features:
        # Dentes: contorno interno da boca
        draw_feature(LANDMARK_INDICES['mouth_inner'])
    
    # Bochechas (região aproximada entre olhos e boca)
    if 'cheeks' in features:
        # Bochecha esquerda: região entre olho esquerdo e boca
        cheek_left_points = np.concatenate([
            landmarks[LANDMARK_INDICES['left_eye']],
            landmarks[[48, 60, 67, 66, 65, 64]]  # Lado esquerdo da boca
        ])
        cheek_left_hull = cv2.convexHull(cheek_left_points.astype(np.int32))
        cv2.fillConvexPoly(mask, cheek_left_hull, 255)
        
        # Bochecha direita: região entre olho direito e boca
        cheek_right_points = np.concatenate([
            landmarks[LANDMARK_INDICES['right_eye']],
            landmarks[[54, 64, 60, 61, 62, 63]]  # Lado direito da boca
        ])
        cheek_right_hull = cv2.convexHull(cheek_right_points.astype(np.int32))
        cv2.fillConvexPoly(mask, cheek_right_hull, 255)
    
    # Testa (região acima das sobrancelhas)
    if 'forehead' in features:
        # Testa: região acima das sobrancelhas até o topo do rosto
        # Usamos os pontos superiores do contorno do rosto e as sobrancelhas
        forehead_points = np.concatenate([
            landmarks[LANDMARK_INDICES['left_eyebrow']],
            landmarks[LANDMARK_INDICES['right_eyebrow']],
            landmarks[[0, 1, 2, 14, 15, 16]]  # Pontos superiores do contorno
        ])
        # Estendemos para cima estimando a altura da testa
        center_x = landmarks[:, 0].mean()
        top_y = min(landmarks[LANDMARK_INDICES['left_eyebrow']][:, 1].min(),
                   landmarks[LANDMARK_INDICES['right_eyebrow']][:, 1].min())
        forehead_height = (landmarks[27, 1] - top_y) * 0.8  # ~80% da distância nariz-sobrancelha
        top_points = np.array([
            [center_x - 50, top_y - forehead_height],
            [center_x + 50, top_y - forehead_height],
        ])
        forehead_points = np.concatenate([forehead_points, top_points])
        forehead_hull = cv2.convexHull(forehead_points.astype(np.int32))
        cv2.fillConvexPoly(mask, forehead_hull, 255)

    # Aplicar dilatação se solicitado
    if dilate_radius > 0:
        kernel = np.ones((dilate_radius * 2 + 1, dilate_radius * 2 + 1), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    
    # Suavização de bordas
    if smooth_edges:
        # Aplicar blur gaussiano suave nas bordas
        mask_blurred = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 1.0)
        # Threshold para manter a forma mas suavizar bordas
        mask = (mask_blurred > 127).astype(np.uint8) * 255
    
    # Aplicar blur adicional se especificado
    if blur_sigma > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), blur_sigma)
        # Re-threshold após blur
        mask = (mask > 127).astype(np.uint8) * 255

    return mask


def visualize_mask(image, mask, alpha=0.5):
    """
    Visualiza a máscara sobreposta na imagem.
    
    Args:
        image (np.ndarray): Imagem original (H, W, 3).
        mask (np.ndarray): Máscara binária (H, W).
        alpha (float): Transparência da sobreposição (0-1).
    
    Returns:
        np.ndarray: Imagem com máscara sobreposta.
    """
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Criar overlay colorido (vermelho para a máscara)
    overlay = image.copy()
    mask_colored = np.zeros_like(image)
    mask_colored[mask > 0] = [0, 0, 255]  # Vermelho
    
    # Combinar imagem e máscara
    result = cv2.addWeighted(overlay, 1 - alpha, mask_colored, alpha, 0)
    
    return result


def get_segmented_regions(landmarks, image_size):
    """
    Retorna todas as regiões segmentadas disponíveis como um dicionário de máscaras.
    
    Args:
        landmarks (np.ndarray): Array de 68 landmarks faciais.
        image_size (tuple): Tamanho da imagem (width, height).
    
    Returns:
        dict: Dicionário com chaves sendo os nomes das regiões e valores sendo as máscaras.
    """
    regions = [
        'eyes', 'left_eye', 'right_eye',
        'eyebrows', 'left_eyebrow', 'right_eyebrow',
        'nose', 'nose_bridge', 'nose_tip', 'nostrils',
        'mouth', 'lips', 'teeth',
        'cheeks', 'forehead'
    ]
    
    masks = {}
    for region in regions:
        masks[region] = get_mask_from_landmarks(
            landmarks, 
            image_size, 
            [region],
            dilate_radius=0,  # Sem dilatação para visualização individual
            smooth_edges=False
        )
    
    return masks
