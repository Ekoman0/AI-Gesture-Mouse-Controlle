import cv2
import numpy as np
import HandTrackingModule as htm
import pyautogui
import time
import threading
import speech_recognition as sr

##########################
wCam, hCam = 640, 480
frameR = 160 # Hassasiyeti artırmak için 100'den 160'a çıkardık (Alan küçüldükçe fare daha çok hızlanır)
smoothening = 6 # Çok ufak daha keskin hissettirmesi için yumuşatmayı 7'den 6'ya çektik
#########################

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.HandDetector(maxHands=1)

wScr, hScr = pyautogui.size()
# Disable failsafe to prevent pyautogui from throwing an exception if the mouse goes to a corner
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0 # Faredeki takılmayı/kasmayı önlemek için her hareket arası gecikmeyi sıfırladık

is_listening = False
voice_typing_active = False
last_fist_time = 0
fist_start_time = 0
win_d_start_time = 0
last_win_d_time = 0

def listen_and_type():
    global is_listening, voice_typing_active
    r = sr.Recognizer()
    
    # sr.Microphone() bilgisayardaki varsayılan mikrofonu (örneğin Camo veya sanal bir mikrofonu) 
    # seçtiği için ses gitmiyordu. Sisteminizdeki "Mikrofon Dizisi (Realtek)" mikrofonunun 
    # index numarası olan 2'yi zorunlu olarak seçiyoruz!
    with sr.Microphone(device_index=2) as source:
        print("Dinleniyor... (Durdurmak için OK işaretini bozun)")
        
        frames = []
        grace_period = 1.0 # 1 tam saniye tolerans!
        last_true_time = time.time()
        
        while True:
            if voice_typing_active:
                last_true_time = time.time()
            elif time.time() - last_true_time > grace_period:
                break 
                
            try:
                # 0.1 saniyelik çok küçük parçalar halinde okuyoruz ki döngü sürekli durumu kontrol edebilsin.
                chunk_audio = r.record(source, duration=0.1)
                frames.append(chunk_audio.get_raw_data())
            except Exception as e:
                pass
                
        print("Kayıt bitti, Google'a gönderiliyor...")
        
        if len(frames) == 0:
            is_listening = False
            return
            
        # Tüm parçaları birleştirip tek bir ses dosyası yapıyoruz
        raw_data = b''.join(frames)
        audio = sr.AudioData(raw_data, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
        
        try:
            text = r.recognize_google(audio, language="tr-TR")
            print(f"Söylenen: {text}")
            
            # Kopyala-Yapıştır mantığı
            import pyperclip
            pyperclip.copy(" " + text)
            pyautogui.hotkey("ctrl", "v")
            
        except sr.UnknownValueError:
            print("Ses anlaşılamadı (Mikrofonunuza ses gitmemiş veya boş kayıt).")
        except Exception as e:
            print(f"Hata: {e}")
            
    time.sleep(0.5) # Hemen tekrar tetiklenmemesi için
    is_listening = False

print(f"Screen Resolution: {wScr}x{hScr}")

while True:
    # 1. Find hand Landmarks
    success, img = cap.read()
    if not success:
        break
    
    # Flip image to act like a mirror
    img = cv2.flip(img, 1)

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    
    # El ekrandan çıkarsa kaydın kapanması için global değişkeni False yapıyoruz
    voice_typing_active = False
    
    # 2. Get the tip of the index and middle fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]   # Index finger tip
        x2, y2 = lmList[12][1:]  # Middle finger tip
        
        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)
        
        # 4. Only Index Finger : Moving Mode
        if fingers[1] == 1 and fingers[2] == 0:
            # 5. Convert Coordinates
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
            
            # 6. Smoothen Values
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening
            
            # 7. Move Mouse
            # Ensure coordinates are within screen bounds
            clocX = max(0, min(wScr, clocX))
            clocY = max(0, min(hScr, clocY))
            
            pyautogui.moveTo(wScr - clocX if False else clocX, clocY) # Image is already flipped
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY
            
        # 8. Both Index and Middle fingers are up : Clicking Mode
        if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0:
            # 9. Find distance between fingers
            length, img, lineInfo = detector.findDistance(8, 12, img)
            
            # 10. Click mouse if distance short
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                pyautogui.click()
                # To prevent multiple clicks very fast, we could add a small delay or cooldown
                time.sleep(0.15)
                
        # 11. Scroll Mode: Index, Middle, Ring fingers up
        if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:
            # We can use the y coordinate of the index finger to determine scroll direction
            # For simplicity, if we move the hand up, scroll up. If down, scroll down.
            # Let's map the y position in the frame to a scroll value.
            # Middle of the active area:
            midY = hCam / 2
            
            if y1 < midY - 30:
                pyautogui.scroll(50) # Scroll up
                cv2.putText(img, "Scroll UP", (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            elif y1 > midY + 30:
                pyautogui.scroll(-50) # Scroll down
                cv2.putText(img, "Scroll DOWN", (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        # 12. Voice Typing Mode: Su Altı OK İşareti
        # Baş(4) ve İşaret(8) parmak uçları birleşik (mesafe < 40), diğer 3 parmak (Orta, Yüzük, Serçe) havada
        length_ok, img, _ = detector.findDistance(4, 8, img, draw=False)
        voice_typing_active = (length_ok < 40 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1)
        
        if voice_typing_active:
            if not is_listening:
                is_listening = True
                threading.Thread(target=listen_and_type, daemon=True).start()
            cv2.putText(img, "Dinleniyor...", (50, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)

        # 13. Fist Mode: Yumruk yapıldığında (Tüm parmaklar kapalı) -> Windows + Tab (Görev Görünümü)
        # MediaPipe baş parmağı kapalı (0) algılamakta zorlanabilir, bu yüzden sadece diğer 4 parmağa bakıyoruz.
        if fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
            if fist_start_time == 0:
                fist_start_time = time.time()
            elif time.time() - fist_start_time > 0.4: # 0.4 saniye tutulursa tetikle
                current_time = time.time()
                if current_time - last_fist_time > 2.0: # Her 2 saniyede maksimum 1 kez tetiklensin (spamı önler)
                    pyautogui.hotkey('win', 'tab')
                    last_fist_time = current_time
                cv2.putText(img, "Gorev Gorunumu (Win+Tab)", (50, 200), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 3)
        else:
            fist_start_time = 0 # Yumruk bozulursa süreyi sıfırla

        # 14. Win+D Mode (Masaüstünü Göster): 3 Parmak Havada
        # Not: Kaydırma(Scroll) işlemi zaten İşaret, Orta ve Yüzük parmağını kullandığı için
        # çakışmamaları adına Masaüstü hareketini "Baş, İşaret ve Orta" parmak olarak ayarlıyoruz. (Diğerleri kapalı)
        if fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
            if win_d_start_time == 0:
                win_d_start_time = time.time()
            elif time.time() - win_d_start_time > 0.4: # 0.4 saniye tutulursa tetikle
                current_time = time.time()
                if current_time - last_win_d_time > 2.0:
                    pyautogui.hotkey('win', 'd')
                    last_win_d_time = current_time
                cv2.putText(img, "Masaustu (Win+D)", (50, 250), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)
        else:
            win_d_start_time = 0

    # 15. Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    
    # 13. Display
    cv2.imshow("Image", img)
    
    # Break loop on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
