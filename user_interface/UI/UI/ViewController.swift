//
//  ViewController.swift
//  UI
//
//  Created by Newsha Seyedi on 2022-05-19.
//

import UIKit
import Foundation

class ViewController: UIViewController, UITextViewDelegate, UITextFieldDelegate {
    let baseURL = String("https://b047-34-133-189-72.ngrok.io/generate_single_image?seed=")
    @IBOutlet weak var image: UIImageView!
    @IBOutlet weak var seed: UITextField!
    override func viewDidLoad() {
        super.viewDidLoad()
        seed.delegate = self
        // Do any additional setup after loading the view.
    }
    @IBOutlet weak var showValue: UILabel!
    func textField(_ textField: UITextField, shouldChangeCharactersIn range: NSRange, replacementString string: String) -> Bool {
        if textField == seed {
                    let allowedCharacters = "1234567890"
                    let allowedCharacterSet = CharacterSet(charactersIn: allowedCharacters)
                    let typedCharacterSet = CharacterSet(charactersIn: string)
                    let alphabet = allowedCharacterSet.isSuperset(of: typedCharacterSet)
                  return alphabet
          }
        return true
      }

    @IBAction func updateValue(_ sender: UISlider) {
        showValue.text = String(format: "%.1f", trunc.value)
    }
    @IBOutlet weak var trunc: UISlider!
    @IBAction func GenerateClick(_ sender: UIButton) {
        print(seed.text!)
        print(trunc.value)
        let url = baseURL + String(seed.text!) + String("&trunc=")+String(trunc.value)
        image.downloaded(from: url)
        image.contentMode = .scaleAspectFit
        
    }
}

