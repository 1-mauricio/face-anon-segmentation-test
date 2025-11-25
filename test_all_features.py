"""
Teste completo de todas as características faciais disponíveis na segmentação.
Este teste verifica todas as opções de segmentação além de olhos e boca.
"""

import numpy as np
import cv2
from PIL import Image
import face_alignment
from diffusers.utils import load_image

from utils.segmentation import get_mask_from_landmarks, visualize_mask
from utils.extractor import get_transform_mat, FaceType
from utils.segmented_anonymization import anonymize_faces_segmented


def test_all_features():
    """Testa todas as características faciais disponíveis."""
    
    print("=" * 70)
    print("TESTE COMPLETO DE TODAS AS CARACTERÍSTICAS FACIAIS")
    print("=" * 70)
    
    # Carregar imagem
    image_path = "my_dataset/test/00482.png"
    print(f"\n1. Carregando imagem: {image_path}")
    
    try:
        original_image = load_image(image_path)
        print(f"   ✓ Imagem carregada: {original_image.size}")
    except Exception as e:
        print(f"   ✗ Erro: {e}")
        return False
    
    # Converter para numpy
    image_np = np.array(original_image)
    if image_np.shape[2] == 4:
        image_np = image_np[:, :, :3]
    
    # Inicializar face alignment
    print("\n2. Inicializando Face Alignment...")
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    
    try:
        fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            face_detector="sfd",
            device=device
        )
        print(f"   ✓ Face Alignment inicializado (device: {device})")
    except Exception as e:
        print(f"   ✗ Erro: {e}")
        return False
    
    # Detectar landmarks
    print("\n3. Detectando landmarks faciais...")
    try:
        preds = fa.get_landmarks(image_np)
        if preds is None or len(preds) == 0:
            print("   ✗ Nenhum rosto detectado")
            return False
        print(f"   ✓ {len(preds)} rosto(s) detectado(s)")
    except Exception as e:
        print(f"   ✗ Erro: {e}")
        return False
    
    # Preparar face alinhada
    face_image_size = 512
    landmarks = preds[0]
    
    mat = get_transform_mat(landmarks, face_image_size, FaceType.WHOLE_FACE)
    face_aligned = cv2.warpAffine(
        image_np,
        mat,
        (face_image_size, face_image_size),
        cv2.INTER_LANCZOS4,
        borderValue=(255, 255, 255),
    )
    
    pts = np.array([landmarks], dtype=np.float32)
    aligned_landmarks = cv2.transform(pts, mat)[0]
    
    print(f"\n4. Testando todas as características faciais...")
    print("=" * 70)
    
    # Definir todos os testes de características
    feature_tests = [
        # Olhos individuais
        {
            'features': ['left_eye'],
            'name': 'Olho Esquerdo',
            'operator': 'blur'
        },
        {
            'features': ['right_eye'],
            'name': 'Olho Direito',
            'operator': 'blur'
        },
        {
            'features': ['eyes'],
            'name': 'Ambos os Olhos',
            'operator': 'blur'
        },
        
        # Sobrancelhas
        {
            'features': ['left_eyebrow'],
            'name': 'Sobrancelha Esquerda',
            'operator': 'blur'
        },
        {
            'features': ['right_eyebrow'],
            'name': 'Sobrancelha Direita',
            'operator': 'blur'
        },
        {
            'features': ['eyebrows'],
            'name': 'Ambas Sobrancelhas',
            'operator': 'blur'
        },
        
        # Nariz
        {
            'features': ['nose_bridge'],
            'name': 'Ponte do Nariz',
            'operator': 'blur'
        },
        {
            'features': ['nose_tip'],
            'name': 'Ponta do Nariz',
            'operator': 'blur'
        },
        {
            'features': ['nostrils'],
            'name': 'Narinas',
            'operator': 'blur'
        },
        {
            'features': ['nose'],
            'name': 'Nariz Completo',
            'operator': 'blur'
        },
        
        # Boca
        {
            'features': ['mouth'],
            'name': 'Boca Completa',
            'operator': 'blur'
        },
        {
            'features': ['lips'],
            'name': 'Lábios',
            'operator': 'blur'
        },
        {
            'features': ['teeth'],
            'name': 'Dentes',
            'operator': 'blur'
        },
        
        # Regiões maiores
        {
            'features': ['cheeks'],
            'name': 'Bochechas',
            'operator': 'mosaic'
        },
        {
            'features': ['forehead'],
            'name': 'Testa',
            'operator': 'mosaic'
        },
        
        # Combinações interessantes
        {
            'features': ['eyes', 'eyebrows'],
            'name': 'Olhos + Sobrancelhas',
            'operator': 'blur'
        },
        {
            'features': ['nose', 'nostrils'],
            'name': 'Nariz + Narinas',
            'operator': 'blur'
        },
        {
            'features': ['lips', 'teeth'],
            'name': 'Lábios + Dentes',
            'operator': 'blur'
        },
        {
            'features': ['eyes', 'nose', 'mouth'],
            'name': 'Olhos + Nariz + Boca',
            'operator': 'mosaic'
        },
        {
            'features': ['eyebrows', 'eyes', 'nose', 'mouth'],
            'name': 'Sobrancelhas + Olhos + Nariz + Boca',
            'operator': 'mosaic'
        },
    ]
    
    results = []
    visualization_images = []
    
    for i, test in enumerate(feature_tests):
        print(f"\n   Teste {i+1}/{len(feature_tests)}: {test['name']}")
        print(f"      Features: {test['features']}")
        
        try:
            # Gerar máscara
            mask = get_mask_from_landmarks(
                aligned_landmarks,
                (face_image_size, face_image_size),
                test['features'],
                dilate_radius=3,
                smooth_edges=True
            )
            
            # Verificar se a máscara não está vazia
            mask_area = np.sum(mask > 0)
            mask_percentage = (mask_area / (face_image_size * face_image_size)) * 100
            
            if mask_area == 0:
                print(f"      ⚠ Máscara vazia - pulando anonimização")
                continue
            
            print(f"      ✓ Máscara gerada ({mask_percentage:.1f}% da face)")
            
            # Visualizar máscara
            vis_image = visualize_mask(face_aligned, mask, alpha=0.5)
            visualization_images.append({
                'image': Image.fromarray(vis_image),
                'name': test['name']
            })
            
            # Aplicar anonimização
            kwargs = {}
            if test['operator'] == 'blur':
                kwargs['kernel_size'] = (31, 31)
            elif test['operator'] == 'mosaic':
                kwargs['block_size'] = 15
            
            anon_image = anonymize_faces_segmented(
                image=original_image,
                face_alignment_model=fa,
                mask_features=test['features'],
                operator_type=test['operator'],
                dilate_radius=3,
                smooth_edges=True,
                **kwargs
            )
            
            # Salvar resultado
            safe_name = test['name'].replace(' ', '_').replace('+', '_').lower()
            output_path = f"test_feature_{i+1:02d}_{safe_name}.png"
            anon_image.save(output_path)
            
            print(f"      ✓ Anonimização concluída")
            print(f"      ✓ Salvo em: {output_path}")
            
            results.append({
                'name': test['name'],
                'features': test['features'],
                'mask_percentage': mask_percentage,
                'output': output_path,
                'status': 'success'
            })
            
        except Exception as e:
            print(f"      ✗ Erro: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'name': test['name'],
                'status': 'error',
                'error': str(e)
            })
    
    # Resumo
    print("\n" + "=" * 70)
    print("RESUMO DOS TESTES")
    print("=" * 70)
    
    successful = [r for r in results if r.get('status') == 'success']
    failed = [r for r in results if r.get('status') == 'error']
    
    print(f"\n✓ Sucessos: {len(successful)}/{len(feature_tests)}")
    print(f"✗ Falhas: {len(failed)}/{len(feature_tests)}")
    
    if successful:
        print("\n✓ Características testadas com sucesso:")
        for r in successful:
            print(f"  - {r['name']}: {r['mask_percentage']:.1f}% da face")
            print(f"    Features: {r['features']}")
            print(f"    Output: {r['output']}\n")
    
    if failed:
        print("\n✗ Erros encontrados:")
        for r in failed:
            print(f"  - {r['name']}: {r.get('error', 'Erro desconhecido')}")
    
    # Criar visualização combinada das máscaras
    if visualization_images:
        print("\n5. Criando visualização combinada das máscaras...")
        try:
            from diffusers.utils import make_image_grid
            
            # Agrupar por categoria
            eye_images = [v for v in visualization_images if 'olho' in v['name'].lower() or 'eye' in v['name'].lower()]
            eyebrow_images = [v for v in visualization_images if 'sobrancelha' in v['name'].lower() or 'eyebrow' in v['name'].lower()]
            nose_images = [v for v in visualization_images if 'nariz' in v['name'].lower() or 'nose' in v['name'].lower()]
            mouth_images = [v for v in visualization_images if 'boca' in v['name'].lower() or 'mouth' in v['name'].lower() or 'lábio' in v['name'].lower() or 'lip' in v['name'].lower() or 'dente' in v['name'].lower() or 'teeth' in v['name'].lower()]
            other_images = [v for v in visualization_images if v not in eye_images + eyebrow_images + nose_images + mouth_images]
            
            all_vis_images = [v['image'] for v in visualization_images[:12]]  # Primeiras 12 para não ficar muito grande
            
            if all_vis_images:
                grid = make_image_grid(all_vis_images, rows=3, cols=4)
                grid.save("test_all_features_masks.png")
                print(f"   ✓ Grid de máscaras salvo em: test_all_features_masks.png")
        except Exception as e:
            print(f"   ⚠ Erro ao criar grid: {e}")
    
    print("\n" + "=" * 70)
    print("✓ TESTE COMPLETO CONCLUÍDO!")
    print("=" * 70)
    
    return len(successful) > 0


if __name__ == "__main__":
    success = test_all_features()
    exit(0 if success else 1)


