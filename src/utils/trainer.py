import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class StyleTrainer:
    def __init__(
        self,
        model,
        style_losses,
        content_losses,
        input_image,
        optimizer,
        style_weight=1,
        content_weight=1000000,
        num_steps=300,
    ):
        self.model = model
        self.input_image = input_image
        self.style_losses = style_losses
        self.content_losses = content_losses
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.num_steps = num_steps
        self.optimizer = optimizer
        self.style_score = 0
        self.content_score = 0

    def run(self):
        step = [0]
        for _ in range(self.num_steps):
            
            def closure():
                # 업데이트 된 입력 이미지의 값을 수정
                with torch.no_grad():
                    self.input_image.clamp_(0, 1)  # 값을 0 ~ 1사이로 CLAMP 및 Inplace

                self.optimizer.zero_grad()
                self.model(self.input_image)
                style_score = 0
                content_score = 0
                
                for style_loss in self.style_losses:
                    style_score += style_loss.loss
                for content_loss in self.content_losses:
                    content_score += content_loss.loss

                style_score *= self.style_weight
                content_score *= self.content_weight
                loss = style_score + content_score
                
                loss.backward()  # 폐쇄 함수 안의 수치만 Gradient
                step[0] += 1
                print(f"steps: {step[0]}, Style Loss: {style_score.item()}, Content Loss: {content_score.item()}")
                
                return style_score + content_score
            self.optimizer.step(closure)

        with torch.no_grad():
            self.input_image.clamp_(0, 1)

    
    def compute_total_loss(self, style_losses, style_score, content_losses, content_score):
        for style_loss in style_losses:
            style_score += style_loss.loss
        for content_loss in content_losses:
            content_score += content_loss.loss

        style_score *= self.style_weight
        content_score *= self.content_weight
        total_loss = style_score + content_score
        return total_loss, style_score, content_score

    def save(self, save_path):
        unloader = transforms.ToPILImage()  # PIL 이미지로 다시 변환
        image = self.input_image.cpu().clone()  # 텐서를 복제하여 변경하지 않음
        image = image.squeeze(0)  # 가짜 배치 차원 제거
        image = unloader(image)
        image.save(save_path, 'png')

    def log_show(self, step, style_score, content_score):
        style_loss = style_score.item()
        content_loss = content_score.item()
        print(f"steps: {step}, Style Loss: {style_loss}, Content Loss: {content_loss}")

    def show(self, title=None):
        unloader = transforms.ToPILImage()  # PIL 이미지로 다시 변환
        plt.ion()
        plt.figure()
        image = self.input_image.cpu().clone()  # 텐서를 복제하여 변경하지 않음
        image = image.squeeze(0)  # 가짜 배치 차원 제거
        image = unloader(image)
        if title is not None:
            plt.title(title)
        plt.imshow(image)
        plt.pause(0.001)  # plots가 업데이트 되도록 잠시 멈춤
        plt.ioff()
        plt.show()
