 AI-Driven COF Design: Image-to-Structure with Inverse Logic Bu proje, Kovalent Organik Kafeslerin (COF) tasarım sürecini hızlandırmak için geliştirilmiş, görüntü işleme ve doğal dil işleme (NLP) tekniklerini birleştiren bir derin öğrenme prototipidir. Sistem, karmaşık kimyasal görselleri veya hedef geometri şemalarını analiz ederek bunları SELFIES ve SMILES formatında dijital verilere dönüştürür.

Öne Çıkan Özellikler
Forward Design (Görsel Okuma): Mevcut molekül çizimlerini saniyeler içinde dijital formüllere (SMILES) çevirir.
Inverse Design (Tersine Tasarım): İçinde molekül bulunmayan boş bir kafes iskeletine (örneğin altıgen bir ızgara) bakarak, bu yapıyı inşa edebilecek en uygun kimyasal yapı taşını tasarlar.
SELFIES Entegrasyonu: SMILES'ın aksine, %100 kimyasal geçerliliğe sahip diziler üreterek hatalı molekül oluşumunu engeller.
Gradio Arayüzü: Teknik bilgisi olmayan kullanıcılar için sürükle-bırak mantığıyla çalışan bir Web arayüzü sunar.
Teknik Mimari
Proje, bir Encoder-Decoder mimarisi üzerine kurulmuştur:Encoder (CNN/Vision): Girdi görselindeki geometrik kısıtları, bağları ve atomik merkezleri yakalar.Decoder (LSTM/RNN): Encoder'dan gelen bilgiyi kullanarak atom bazında bir kimyasal dizi (SELFIES) inşa eder.Post-Processing: Üretilen SELFIES kodunu, endüstri standardı olan SMILES formatına tercüme eder.
Kullanılan Teknolojiler 
Derin Öğrenme: PyTorchKimya Bilişimi: RDKit, SELFIESVeri Görselleştirme: Matplotlib, PILArayüz: GradioDil: Python 3.10+ BaşlangıçGereksinimlerBashpip install torch rdkit selfies gradio matplotlib pillow
Kullanım 
Modeli eğitin veya önceden eğitilmiş ağırlıkları (.pth) yükleyin.demo.launch() hücresini çalıştırarak yerel sunucuyu başlatın.Bir molekül resmi veya hedef geometri şeması yükleyerek sonucu gözlemleyin.
Örnek Sonuçlar
Girdi Tipi Amaç Çıktı (SMILES)Molekül Görseli Dijital Arşivleme (Forward)Nc1ccc(cc1)-c2ccc(N)cc2Boş Altıgen Şema Malzeme Tasarımı (Inverse)C1(C#CC2...C=C2)=CC(C#CC3...C=C3)=CC...
Bu bir araştırma prototipidir. Veri seti çeşitliliğini artırmak veya farklı topolojiler (kare, üçgen kafesler) eklemek için PR göndermekten çekinmeyin!
