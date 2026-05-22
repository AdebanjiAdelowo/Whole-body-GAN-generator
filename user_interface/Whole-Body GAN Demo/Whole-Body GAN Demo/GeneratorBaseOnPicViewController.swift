//
//  GeneratorBaseOnPicViewController.swift
//  Whole-Body GAN Demo
//
//  Created by Newsha Seyedi on 2022-05-24.
//

import UIKit
import Foundation
import AVKit
import AVFoundation
import Firebase

class GeneratorBaseOnPicViewController: UIViewController, UITextViewDelegate, UITextFieldDelegate, UINavigationControllerDelegate, UIImagePickerControllerDelegate {

    
    @IBOutlet weak var b1: UIButton!
    @IBOutlet weak var b2: UIButton!
    @IBOutlet weak var b3: UIButton!
    @IBOutlet weak var step: UISlider!
    @IBOutlet weak var seed: UITextField!
    @IBOutlet weak var imageFrame: UIImageView!
    @IBOutlet weak var stepShow: UITextField!
    @IBOutlet weak var but: UIButton!
    @IBOutlet weak var progressBar: UIProgressView!
    @IBOutlet weak var progLabel: UILabel!
    let maxSteps = Int(1000)
    
    override func viewDidLoad() {
        super.viewDidLoad()
        seed.delegate = self
        stepShow.text = "1"
        stepShow.delegate = self
        step.maximumValue = Float(maxSteps)
        imageView.isHidden = true
        progressBar.isHidden = true
        progLabel.isHidden = true
        let tap = UITapGestureRecognizer(target: self, action: #selector(UIInputViewController.dismissKeyboard))
        view.addGestureRecognizer(tap)
        title = "Whole Body Generator"
    }
    
    @IBAction func generate(_ sender: Any) {
        let url = URL(string: "https://file-examples.com/storage/fe2a923118628f9c0986d27/2017/04/file_example_MP4_480_1_5MG.mp4")
        let player = AVPlayer(url: url!)
        let avpController = AVPlayerViewController()
        avpController.player = player
        self.present(avpController, animated: true) {
            avpController.player!.play()
        }
    }
    
    
    @IBAction func stepChange(_ sender: Any) {
        stepShow.text = String(Int(step.value))
    }
    
    @objc func dismissKeyboard() {
        //Causes the view (or one of its embedded text fields) to resign the first responder status.
        view.endEditing(true)
//        relax()
    }
    

    @IBAction func stepShowChange(_ sender: Any) {
        relax()
    }
    
    func relax() {
        print(stepShow.text as Any)
        if (stepShow.text == "") {
            return
        }
        let stp = Int(stepShow.text!) ?? 0
        if stp < 1 || stp > maxSteps {
            stepShow.text = String(Int(step.value))
        } else {
            step.value = Float(stp)
        }
    }
    
    func textField(_ textField: UITextField, shouldChangeCharactersIn range: NSRange, replacementString string: String) -> Bool {
        
        if textField == seed || textField == stepShow{
            let allowedCharacters = "1234567890"
            let allowedCharacterSet = CharacterSet(charactersIn: allowedCharacters)
            let typedCharacterSet = CharacterSet(charactersIn: string)
            let alphabet = allowedCharacterSet.isSuperset(of: typedCharacterSet)
                return alphabet
        }
        return true
      }
    
    @IBAction func takePhoto(_ sender: Any) {
        let vc = UIImagePickerController()
        vc.sourceType = .camera
        vc.allowsEditing = true
        vc.delegate = self
        present(vc, animated: true)
    }
    
    @IBAction func loadPicFromLib(_ sender: Any) {
        let vc = UIImagePickerController()
        vc.sourceType = .photoLibrary
        vc.allowsEditing = true
        vc.delegate = self
        present(vc, animated: true)
    }
    
    
    @IBOutlet weak var imageView: UIImageView!
    
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        picker.dismiss(animated: true, completion: nil)

        guard let image = info[.editedImage] as? UIImage else {
            print("No image found")
            return
        }

        // print out the image size as a test
        print(image.size)
        
        let uploadRef = Storage.storage().reference(withPath: "image.jpg")
        guard let imageData = image.jpegData(compressionQuality: 1) else {
            print("Cannot convert to jpeg.")
            return
        }
        let uploadMeta = StorageMetadata.init()
        uploadMeta.contentType = "image/jpeg"
        progressBar.isHidden = false
        progLabel.isHidden = false
        b1.isEnabled = false
        b2.isEnabled = false
        b3.isEnabled = false
        imageView.isHidden = true
        let taskRef = uploadRef.putData(imageData, metadata: uploadMeta) {(downloadMetaData, error) in
            if let error = error {
                print("Cannot upload!")
                return
            }
            print("Download Complete!")
        }
        taskRef.observe(.progress, handler: {[weak self](snapshot) in
            guard let pct = snapshot.progress?.fractionCompleted else {
                return
            }
            self!.progressBar.progress = Float(pct)
            let tmp = String(Int(Float(pct) * 100))
            let prc = tmp + String("%")
            self!.progLabel.text = prc
        })
        taskRef.observe(.success, handler: {[weak self](snapshot) in
            self!.imageFrame.image = image
            self!.progressBar.isHidden = true
            self!.progLabel.isHidden = true
            self!.progLabel.text = "0%"
            self!.progressBar.progress = Float(0)
//            self!.imageView.isHidden = false
            self!.b1.isEnabled = true
            self!.b2.isEnabled = true
            self!.b3.isEnabled = true
        })
        
    }
    
    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        picker.dismiss(animated: true, completion: nil)
    }
    
}
