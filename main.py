import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import math
from skimage.feature import graycomatrix, graycoprops

class ImageFeatureExtractor:
    def __init__(self, base_folder='dataset'):
        self.base_folder = base_folder
        self.color_results = []
        self.shape_results = []
        self.color_moments_results = []
        self.texture_results = []
        self.glcm_results = []
        self.edge_results = []
    
    def get_dominant_color_name(self, hue_value):
        """Konversi nilai Hue ke nama warna"""
        if hue_value < 10 or hue_value > 170:
            return "Merah"
        elif 10 <= hue_value < 25:
            return "Oranye"
        elif 25 <= hue_value < 35:
            return "Kuning"
        elif 35 <= hue_value < 85:
            return "Hijau"
        elif 85 <= hue_value < 130:
            return "Biru"
        elif 130 <= hue_value < 155:
            return "Ungu / Pink"
        else:
            return "Tidak dikenal"
    
    def extract_dominant_colors(self, image):
        """Ekstraksi warna dominan dari gambar"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        height, width = hsv.shape[:2]
        if height > 300 or width > 300:
            scale = 300 / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            hsv = cv2.resize(hsv, (new_width, new_height))
        
        hue_channel = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        
        mask = (saturation > 30) & (value > 30)
        
        if np.sum(mask) == 0:
            filtered_hue = hue_channel.flatten()
        else:
            filtered_hue = hue_channel[mask]
        
        # Hitung distribusi warna menggunakan histogram
        hue_hist, _ = np.histogram(filtered_hue, bins=180, range=(0, 180))
        
        # Ambil top 10 hue values berdasarkan frekuensi
        top_indices = np.argsort(hue_hist)[-10:][::-1]
        most_common = [(int(idx), int(hue_hist[idx])) for idx in top_indices if hue_hist[idx] > 0]
        
        total_pixels = sum([count for _, count in most_common])
        weighted_hue = sum([hue * count for hue, count in most_common]) / total_pixels if total_pixels > 0 else 0
        
        colors = []
        for hue, count in most_common[:5]:
            color_name = self.get_dominant_color_name(hue)
            colors.append((color_name, hue, count))
        
        color_summary = {}
        for color_name, hue, count in colors:
            if color_name not in color_summary:
                color_summary[color_name] = {'count': 0, 'hues': []}
            color_summary[color_name]['count'] += count
            color_summary[color_name]['hues'].append(hue)
        
        sorted_colors = sorted(color_summary.items(), key=lambda x: x[1]['count'], reverse=True)
        
        result = {}
        for idx, (color_name, data) in enumerate(sorted_colors, 1):
            avg_hue = np.mean(data['hues'])
            result[f'Hue_{idx}'] = round(avg_hue, 2)
            result[f'Warna_{idx}'] = color_name
        
        return result
    
    def extract_shape_features(self, image):
        """Ekstraksi fitur bentuk dari gambar"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if perimeter > 0:
                circularity = (4 * math.pi * area) / (perimeter ** 2)
            else:
                circularity = 0
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            rect_area = w * h
            extent = area / rect_area if rect_area > 0 else 0
            
            return {
                'Area': round(area, 1),
                'Perimeter': round(perimeter, 1),
                'Circularity': round(circularity, 16),
                'Aspect_Ratio': round(aspect_ratio, 2),
                'Solidity': round(solidity, 16),
                'Extent': round(extent, 16)
            }
        else:
            return {
                'Area': 0,
                'Perimeter': 0,
                'Circularity': 0,
                'Aspect_Ratio': 0,
                'Solidity': 0,
                'Extent': 0
            }
    
    def extract_color_moments(self, image):
        """Ekstraksi color moments (Mean RGB, Mean HSV, Std RGB, Std HSV)"""
        # RGB moments
        b, g, r = cv2.split(image)
        mean_r = np.mean(r)
        mean_g = np.mean(g)
        mean_b = np.mean(b)
        std_r = np.std(r)
        std_g = np.std(g)
        std_b = np.std(b)
        
        # HSV moments
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        mean_h = np.mean(h)
        mean_s = np.mean(s)
        mean_v = np.mean(v)
        std_h = np.std(h)
        std_s = np.std(s)
        std_v = np.std(v)
        
        # Skewness dan Kurtosis
        from scipy import stats
        skew_r = stats.skew(r.flatten())
        skew_g = stats.skew(g.flatten())
        skew_b = stats.skew(b.flatten())
        kurt_r = stats.kurtosis(r.flatten())
        kurt_g = stats.kurtosis(g.flatten())
        kurt_b = stats.kurtosis(b.flatten())
        
        return {
            'Mean_R': round(mean_r, 2),
            'Mean_G': round(mean_g, 2),
            'Mean_B': round(mean_b, 2),
            'Std_R': round(std_r, 2),
            'Std_G': round(std_g, 2),
            'Std_B': round(std_b, 2),
            'Mean_H': round(mean_h, 2),
            'Mean_S': round(mean_s, 2),
            'Mean_V': round(mean_v, 2),
            'Std_H': round(std_h, 2),
            'Std_S': round(std_s, 2),
            'Std_V': round(std_v, 2),
            'Skew_R': round(skew_r, 2),
            'Skew_G': round(skew_g, 2),
            'Skew_B': round(skew_b, 2),
            'Kurt_R': round(kurt_r, 2),
            'Kurt_G': round(kurt_g, 2),
            'Kurt_B': round(kurt_b, 2)
        }
    
    def extract_texture_features(self, image):
        """Ekstraksi fitur tekstur (contrast, correlation, energy, homogeneity, entropy)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        mean = np.mean(gray)
        std = np.std(gray)
        
        contrast = std ** 2
        correlation = std / mean if mean > 0 else 0
        
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        hist = hist / hist.sum()
        energy = np.sum(hist ** 2)
        
        homogeneity = np.sum(hist / (1 + np.arange(256)))
        dissimilarity = np.std(gray)
        
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist))
        
        # FFT features
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.abs(fshift)
        fft_mean = np.mean(magnitude_spectrum)
        fft_std = np.std(magnitude_spectrum)
        
        # Gabor features (simplified)
        gabor_mean = np.mean(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
        
        return {
            'Contrast': round(contrast, 2),
            'Correlation': round(correlation, 16),
            'Energy': round(energy, 16),
            'Homogeneity': round(homogeneity, 2),
            'Dissimilarity': round(dissimilarity, 2),
            'Entropy': round(entropy, 2),
            'FFT_Mean': round(fft_mean, 2),
            'FFT_Std': round(fft_std, 2),
            'Gabor_Mean': round(gabor_mean, 2)
        }
    
    def extract_glcm_features(self, image):
        """Ekstraksi fitur GLCM (Gray Level Co-occurrence Matrix)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize jika gambar terlalu besar untuk efisiensi
        if gray.shape[0] > 512 or gray.shape[1] > 512:
            scale = 512 / max(gray.shape)
            new_size = (int(gray.shape[1] * scale), int(gray.shape[0] * scale))
            gray = cv2.resize(gray, new_size)
        
        # Kurangi level gray menjadi 256 -> 32 untuk efisiensi
        gray = (gray // 8).astype(np.uint8)
        
        # Hitung GLCM untuk 4 arah: 0°, 45°, 90°, 135°
        distances = [1]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        glcm = graycomatrix(gray, distances=distances, angles=angles, 
                           levels=32, symmetric=True, normed=True)
        
        # Ekstrak properti GLCM
        contrast = graycoprops(glcm, 'contrast')[0]
        dissimilarity = graycoprops(glcm, 'dissimilarity')[0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0]
        energy = graycoprops(glcm, 'energy')[0]
        correlation = graycoprops(glcm, 'correlation')[0]
        asm = graycoprops(glcm, 'ASM')[0]
        
        # Rata-rata dari 4 arah
        result = {
            'GLCM_Contrast': round(np.mean(contrast), 4),
            'GLCM_Dissimilarity': round(np.mean(dissimilarity), 4),
            'GLCM_Homogeneity': round(np.mean(homogeneity), 4),
            'GLCM_Energy': round(np.mean(energy), 4),
            'GLCM_Correlation': round(np.mean(correlation), 4),
            'GLCM_ASM': round(np.mean(asm), 4)
        }
        
        # Tambahkan per arah (opsional, untuk analisis lebih detail)
        for i, angle in enumerate(['0deg', '45deg', '90deg', '135deg']):
            result[f'GLCM_Contrast_{angle}'] = round(contrast[i], 4)
            result[f'GLCM_Homogeneity_{angle}'] = round(homogeneity[i], 4)
            result[f'GLCM_Energy_{angle}'] = round(energy[i], 4)
        
        return result
    
    def extract_edge_features(self, image):
        """Ekstraksi fitur deteksi tepi menggunakan Canny Edge Density"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Terapkan Gaussian Blur untuk mengurangi noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        
        # Deteksi tepi dengan Canny (threshold rendah dan tinggi)
        edges_low = cv2.Canny(blurred, 50, 150)
        edges_medium = cv2.Canny(blurred, 100, 200)
        edges_high = cv2.Canny(blurred, 150, 250)
        
        # Hitung edge density (persentase piksel edge terhadap total piksel)
        total_pixels = gray.shape[0] * gray.shape[1]
        
        edge_density_low = (np.count_nonzero(edges_low) / total_pixels) * 100
        edge_density_medium = (np.count_nonzero(edges_medium) / total_pixels) * 100
        edge_density_high = (np.count_nonzero(edges_high) / total_pixels) * 100
        
        # Hitung jumlah kontur dari edge
        contours_low, _ = cv2.findContours(edges_low, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours_medium, _ = cv2.findContours(edges_medium, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours_high, _ = cv2.findContours(edges_high, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Hitung rata-rata panjang kontur
        avg_contour_length_low = np.mean([cv2.arcLength(c, True) for c in contours_low]) if contours_low else 0
        avg_contour_length_medium = np.mean([cv2.arcLength(c, True) for c in contours_medium]) if contours_medium else 0
        avg_contour_length_high = np.mean([cv2.arcLength(c, True) for c in contours_high]) if contours_high else 0
        
        # Deteksi tepi menggunakan Sobel (untuk perbandingan)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        sobel_mean = np.mean(sobel_magnitude)
        sobel_std = np.std(sobel_magnitude)
        
        return {
            'Canny_Edge_Density_Low': round(edge_density_low, 4),
            'Canny_Edge_Density_Medium': round(edge_density_medium, 4),
            'Canny_Edge_Density_High': round(edge_density_high, 4),
            'Canny_Contour_Count_Low': len(contours_low),
            'Canny_Contour_Count_Medium': len(contours_medium),
            'Canny_Contour_Count_High': len(contours_high),
            'Canny_Avg_Contour_Length_Low': round(avg_contour_length_low, 2),
            'Canny_Avg_Contour_Length_Medium': round(avg_contour_length_medium, 2),
            'Canny_Avg_Contour_Length_High': round(avg_contour_length_high, 2),
            'Sobel_Mean': round(sobel_mean, 2),
            'Sobel_Std': round(sobel_std, 2)
        }
    
    def process_image(self, image_path):
        """Proses satu gambar dan ekstrak semua fitur"""
        try:
            image = cv2.imread(str(image_path))
            
            if image is None:
                print(f"Gagal membaca gambar: {image_path}")
                return None, None, None, None, None, None
            
            nama_file = image_path.name
            
            # Ekstraksi warna dominan
            color_result = {'Nama_File': nama_file}
            color_result.update(self.extract_dominant_colors(image))
            
            # Ekstraksi bentuk
            shape_result = {'Nama_File': nama_file}
            shape_result.update(self.extract_shape_features(image))
            
            # Ekstraksi color moments
            color_moments_result = {'Nama_File': nama_file}
            color_moments_result.update(self.extract_color_moments(image))
            
            # Ekstraksi tekstur
            texture_result = {'Nama_File': nama_file}
            texture_result.update(self.extract_texture_features(image))
            
            # Ekstraksi GLCM
            glcm_result = {'Nama_File': nama_file}
            glcm_result.update(self.extract_glcm_features(image))
            
            # Ekstraksi edge features
            edge_result = {'Nama_File': nama_file}
            edge_result.update(self.extract_edge_features(image))
            
            return color_result, shape_result, color_moments_result, texture_result, glcm_result, edge_result
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None, None, None, None, None, None
    
    def process_category(self, category_path):
        """Proses semua gambar dalam satu kategori"""
        category_name = category_path.name
        print(f"\nMemproses kategori: {category_name}")
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(category_path.glob(f'*{ext}')))
            image_files.extend(list(category_path.glob(f'*{ext.upper()}')))
        
        image_files.sort()
        
        print(f"Ditemukan {len(image_files)} gambar")
        
        color_results = []
        shape_results = []
        color_moments_results = []
        texture_results = []
        glcm_results = []
        edge_results = []
        
        for idx, image_path in enumerate(image_files, 1):
            print(f"Processing {idx}/{len(image_files)}: {image_path.name}")
            color, shape, moments, texture, glcm, edge = self.process_image(image_path)
            
            if all([color, shape, moments, texture, glcm, edge]):
                color_results.append(color)
                shape_results.append(shape)
                color_moments_results.append(moments)
                texture_results.append(texture)
                glcm_results.append(glcm)
                edge_results.append(edge)
        
        return {
            'color': color_results,
            'shape': shape_results,
            'color_moments': color_moments_results,
            'texture': texture_results,
            'glcm': glcm_results,
            'edge': edge_results
        }, category_name
    
    def save_to_csv(self, results, category_name, output_type, output_folder):
        """Simpan hasil ke CSV"""
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(results)
        
        # Tentukan nama file berdasarkan tipe output
        csv_filename = output_path / f'hasil_{output_type}_{category_name}.csv'
        
        df.to_csv(csv_filename, index=False)
        print(f"✓ Hasil disimpan ke: {csv_filename}")
        
        print(f"  Preview (3 baris pertama):")
        print(df.head(3).to_string(index=False))
        print()
        
        return csv_filename
    
    def run(self):
        """Jalankan ekstraksi untuk semua kategori"""
        base_path = Path(self.base_folder)
        
        if not base_path.exists():
            print(f"Error: Folder '{self.base_folder}' tidak ditemukan!")
            print("Silakan buat folder dengan struktur:")
            print("dataset/")
            print("  ├── apel/")
            print("  ├── nanas/")
            print("  └── pisang/")
            return
        
        categories = [f for f in base_path.iterdir() if f.is_dir()]
        
        if not categories:
            print(f"Tidak ada folder kategori dalam '{self.base_folder}'")
            return
        
        print(f"Ditemukan {len(categories)} kategori")
        
        # Dictionary untuk menyimpan semua hasil gabungan
        all_results = {
            'color': [],
            'shape': [],
            'color_moments': [],
            'texture': [],
            'glcm': [],
            'edge': [],
            'combined': []  # Untuk gabungan semua fitur
        }
        
        for category_path in categories:
            results_dict, category_name = self.process_category(category_path)
            
            # Buat folder output untuk kategori ini
            category_output_folder = Path('results') / category_name
            
            if results_dict['color']:
                print(f"\n{'='*60}")
                print(f"Menyimpan hasil untuk kategori: {category_name}")
                print(f"Output folder: {category_output_folder}")
                print(f"{'='*60}\n")
                
                # Simpan setiap jenis ekstraksi ke CSV terpisah per kategori
                self.save_to_csv(results_dict['color'], category_name, 'warna', category_output_folder)
                self.save_to_csv(results_dict['shape'], category_name, 'bentuk', category_output_folder)
                self.save_to_csv(results_dict['color_moments'], category_name, 'color_moments', category_output_folder)
                self.save_to_csv(results_dict['texture'], category_name, 'tekstur', category_output_folder)
                self.save_to_csv(results_dict['glcm'], category_name, 'glcm', category_output_folder)
                self.save_to_csv(results_dict['edge'], category_name, 'edge', category_output_folder)
                
                # Tambahkan label kategori ke setiap hasil
                for result_type in ['color', 'shape', 'color_moments', 'texture', 'glcm', 'edge']:
                    for item in results_dict[result_type]:
                        item_with_label = {'Kategori': category_name, **item}
                        all_results[result_type].append(item_with_label)
                
                # Gabungkan semua fitur untuk setiap gambar
                for i in range(len(results_dict['color'])):
                    combined_item = {
                        'Kategori': category_name,
                        'Nama_File': results_dict['color'][i]['Nama_File']
                    }
                    # Gabungkan semua fitur kecuali Nama_File yang duplikat
                    for result_type in ['color', 'shape', 'color_moments', 'texture', 'glcm', 'edge']:
                        for key, value in results_dict[result_type][i].items():
                            if key != 'Nama_File':
                                combined_item[key] = value
                    
                    all_results['combined'].append(combined_item)
                
                print(f"✓ Selesai memproses {len(results_dict['color'])} gambar dari kategori {category_name}\n")
            else:
                print(f"Tidak ada hasil untuk kategori {category_name}")
        
        # Simpan file gabungan semua kategori
        if all_results['color']:
            print("\n" + "="*60)
            print("Membuat file gabungan SEMUA KATEGORI...")
            print("="*60 + "\n")
            
            # Folder untuk file gabungan semua kategori
            semua_folder = Path('results') / 'semua'
            
            # Simpan setiap jenis fitur gabungan ke folder 'semua'
            self.save_to_csv(all_results['color'], 'semua_kategori', 'warna', semua_folder)
            self.save_to_csv(all_results['shape'], 'semua_kategori', 'bentuk', semua_folder)
            self.save_to_csv(all_results['color_moments'], 'semua_kategori', 'color_moments', semua_folder)
            self.save_to_csv(all_results['texture'], 'semua_kategori', 'tekstur', semua_folder)
            self.save_to_csv(all_results['glcm'], 'semua_kategori', 'glcm', semua_folder)
            self.save_to_csv(all_results['edge'], 'semua_kategori', 'edge', semua_folder)
            
            # Simpan file gabungan SEMUA FITUR langsung di folder results
            print(f"\n{'='*60}")
            print("Membuat file GABUNGAN LENGKAP (semua fitur + semua kategori)...")
            print(f"{'='*60}\n")
            
            results_folder = Path('results')
            df_combined = pd.DataFrame(all_results['combined'])
            csv_filename = results_folder / 'hasil_semua_fitur_buah.csv'
            df_combined.to_csv(csv_filename, index=False)
            
            print(f"✓ Hasil disimpan ke: {csv_filename}")
            print(f"  Total kolom: {len(df_combined.columns)}")
            print(f"  Total baris: {len(df_combined)}")
            print(f"\n  Preview (3 baris pertama, beberapa kolom):")
            preview_cols = ['Kategori', 'Nama_File'] + list(df_combined.columns[2:7])
            print(df_combined[preview_cols].head(3).to_string(index=False))
            print(f"  ... dan {len(df_combined.columns) - len(preview_cols)} kolom lainnya")
            print()
            
            print(f"\n✓ Total {len(all_results['combined'])} gambar dari semua kategori berhasil diproses!")
        
        print("\n" + "="*60)
        print("=== PROSES SELESAI ===")
        print("\nStruktur output yang dihasilkan:")
        print("results/")
        print("  ├── hasil_semua_fitur_buah.csv  ← ⭐ GABUNGAN LENGKAP SEMUA")
        print("  ├── semua/")
        print("  │   ├── hasil_warna_semua_kategori.csv")
        print("  │   ├── hasil_bentuk_semua_kategori.csv")
        print("  │   ├── hasil_color_moments_semua_kategori.csv")
        print("  │   ├── hasil_tekstur_semua_kategori.csv")
        print("  │   ├── hasil_glcm_semua_kategori.csv")
        print("  │   └── hasil_edge_semua_kategori.csv")
        for category_path in categories:
            category_name = category_path.name
            print(f"  ├── {category_name}/")
            print(f"  │   ├── hasil_warna_{category_name}.csv")
            print(f"  │   ├── hasil_bentuk_{category_name}.csv")
            print(f"  │   ├── hasil_color_moments_{category_name}.csv")
            print(f"  │   ├── hasil_tekstur_{category_name}.csv")
            print(f"  │   ├── hasil_glcm_{category_name}.csv")
            print(f"  │   └── hasil_edge_{category_name}.csv")

# Main program
if __name__ == "__main__":
    print("="*60)
    print("  PROGRAM EKSTRAKSI FITUR GAMBAR BUAH")
    print("  dengan GLCM dan Canny Edge Detection")
    print("="*60)
    print("\nPastikan struktur folder seperti berikut:")
    print("dataset/")
    print("  ├── apel/")
    print("  ├── nanas/")
    print("  └── pisang/")
    print("\nMemulai proses ekstraksi...\n")
    
    extractor = ImageFeatureExtractor(base_folder='dataset')
    extractor.run()
    
    print("\n" + "="*60)
    print("Program selesai dijalankan!")
    print("="*60)