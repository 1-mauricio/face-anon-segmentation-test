"""
Teste completo da anonimização segmentada (sem modelos de difusão).
Testa apenas blur e mosaic para verificação rápida.
"""

import numpy as np
from PIL import Image
import face_alignment
from diffusers.utils import load_image

from utils.segmented_anonymization import anonymize_faces_segmented


def test_anonymization():
    """Testa a anonimização segmentada com blur e mosaic."""
    
    print("=" * 60)
    print("TESTE DE ANONIMIZAÇÃO SEGMENTADA")
    print("=" * 60)
    
    # Carregar imagem
    image_path = "my_dataset/test/14795.png"
    print(f"\n1. Carregando imagem: {image_path}")
    
    try:
        original_image = load_image(image_path)
        print(f"   ✓ Imagem carregada: {original_image.size}")
    except Exception as e:
        print(f"   ✗ Erro: {e}")
        return False
    
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
    
    # Testar diferentes combinações de máscaras e operadores
    test_cases = [
        {
            'features': ['eyes'],
            'operator': 'blur',
            'name': 'Blur - Apenas Olhos'
        },
        {
            'features': ['mouth'],
            'operator': 'blur',
            'name': 'Blur - Apenas Boca'
        },
        {
            'features': ['eyes', 'mouth'],
            'operator': 'blur',
            'name': 'Blur - Olhos + Boca'
        },
        {
            'features': ['eyes', 'mouth'],
            'operator': 'mosaic',
            'name': 'Mosaic - Olhos + Boca'
        },
        {
            'features': ['eyes', 'mouth', 'nostrils'],
            'operator': 'mosaic',
            'name': 'Mosaic - Olhos + Boca + Narinas'
        },
    ]
    
    print("\n3. Testando anonimização segmentada...")
    results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"\n   Teste {i+1}/{len(test_cases)}: {test_case['name']}")
        print(f"      Features: {test_case['features']}")
        print(f"      Operator: {test_case['operator']}")
        
        try:
            # Parâmetros específicos por operador
            kwargs = {}
            if test_case['operator'] == 'blur':
                kwargs['kernel_size'] = (31, 31)
            elif test_case['operator'] == 'mosaic':
                kwargs['block_size'] = 15
            
            anon_image = anonymize_faces_segmented(
                image=original_image,
                face_alignment_model=fa,
                mask_features=test_case['features'],
                operator_type=test_case['operator'],
                dilate_radius=3,
                smooth_edges=True,
                **kwargs
            )
            
            # Salvar resultado
            output_path = f"test_anon_{i+1}_{test_case['operator']}.png"
            anon_image.save(output_path)
            print(f"      ✓ Anonimização concluída")
            print(f"      ✓ Resultado salvo em: {output_path}")
            
            results.append({
                'name': test_case['name'],
                'image': anon_image,
                'path': output_path
            })
            
        except Exception as e:
            print(f"      ✗ Erro: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"✓ TESTE CONCLUÍDO: {len(results)}/{len(test_cases)} casos de teste passaram")
    print("=" * 60)
    
    if len(results) > 0:
        print("\nImagens geradas:")
        for r in results:
            print(f"  - {r['path']}")
    
    return len(results) == len(test_cases)


if __name__ == "__main__":
    success = test_anonymization()
    exit(0 if success else 1)


