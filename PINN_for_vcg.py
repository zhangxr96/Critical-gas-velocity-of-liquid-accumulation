# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 20:00:20 2024

@author: ZhangXR_CUP
"""

#%% 导入支持模块
import torch
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import math

#%% 自定义激活函数
class Activation(torch.nn.Module):
    def __init__(self, n):
        super(Activation, self).__init__()
        self.n = n

    def forward(self, x):
        return torch.min(torch.max(x, torch.tensor(0.0)), torch.tensor(self.n))

#%% PGNN的建立
class PGNN(nn.Module):
    # 模型结构
    def __init__(self, layers):
        super(PGNN, self).__init__()
        self.layers = layers
        self.activation_tan = nn.Tanh()
        self.activation_relu = nn.ReLU()
        self.phi_linears = nn.ModuleList()
        self.ug_linears = nn.ModuleList()
        self.phi_activation = Activation(0.8*3.14)
        
        for i in range(len(layers) - 1):
            self.phi_linears.append(nn.Linear(layers[i], layers[i+1]))
            self.ug_linears.append(nn.Linear(layers[i], layers[i+1]))
        for i in range(len(layers) - 1):
            self.phi_linears[i].weight.data.fill_(0.15) # 将所有权值设为0.5
            self.ug_linears[i].weight.data.fill_(0.2) # 将所有权值设为0.5
            # set biases to zero
            nn.init.zeros_(self.phi_linears[i].bias.data)
            nn.init.zeros_(self.ug_linears[i].bias.data)
            
        self.criterion = torch.nn.MSELoss(reduction='mean')
    # 计算
    def forward(self, x):
        if torch.is_tensor(x) != True:
            x = torch.tensor(x, dtype=torch.float32)
        
        ug = x.to(torch.float32)
        phi = x.to(torch.float32)
        
        for i in range(len(self.layers) - 2):
            ug = self.activation_relu(self.ug_linears[i](ug))
            phi = self.phi_activation(self.phi_linears[i](phi))
            
        phi = self.phi_activation(self.phi_linears[-1](phi))
        ug = nn.functional.relu(self.ug_linears[-1](ug))
        return phi, ug
  
    # 最小界面剪切应力
    def loss_inter_tau(self, phi, u_g, mu_l, rho_l, rho_g, mu_g, u_sl, angel, r):
        Sg = 2*3.14*r * (2*3.14 - 2*phi)/(2*3.14)
        Si = 2 * torch.sin(phi) * r
        Sl = 2*3.14*r * 2*phi/(2*3.14)
        Al = r*r*(phi - torch.sin(phi)*torch.cos(phi))
        Hl = Al/math.pi/r**2
        Ag = (1 - Hl) * 3.14 * r**2
        Dg = 4 * Ag / (Sg + Si)
        Dl = 4 * Al / (Sl + Si)
        u_l = u_sl / Hl
        Re_g = rho_g * u_g * Dg / mu_g
        Re_l = rho_l * u_l * Dl / mu_l
        fg = 0.046*Re_g**(-0.2)
        fl = 16*Re_l**(-1)
        tau_wg = fg*rho_g*u_g**2/2
        tau_wl = fl*rho_l*u_l**2/2
        F_value = 0.04755725830265764*phi + 0.0017869984535360073
        phi_B = 1/3.14*(phi - torch.sin(2*phi) + 1/4*torch.sin(4*phi)) + 2*torch.sin(2*phi)**2*F_value
        phi_i = 4/3*torch.sin(phi)**2*(torch.sin(phi)/3.14 - 6*F_value*torch.cos(phi))
        tau_i = 8*mu_l*u_sl/r/(2*phi_i + r*phi_B*Si/Ag) + r*phi_B/(2*phi_i + r*phi_B*Si/Ag)*((rho_l - rho_g)*9.8*torch.sin(angel) - tau_wg*Sg/Ag)
        
        Fr = rho_g * u_g**2 / (rho_l - rho_g) / 9.81 / r / 2 / math.cos(angel)
        fi = (1 + 5.66*(Hl*Fr*Re_g**0.2)**0.25) * fg
        tau_i_ = (fi*rho_g*(u_g - u_l)*abs(u_g - u_l)/2)
        
        if tau_i < 0:
            loss_tau = torch.abs(tau_i)*100
        else:
            loss_tau = torch.abs(tau_i)
        return loss_tau, torch.abs((tau_i_-tau_i)/tau_i-1)
    
    # 零壁面剪切应力
    def loss_wall(self, phi, u_g, mu_l, rho_l, rho_g, mu_g, u_sl, angel, r):
        Sg = 2*3.14*r * (2*3.14 - 2*phi)/(2*3.14)
        Si = 2 * torch.sin(phi) * r
        Sl = 2*3.14*r * 2*phi/(2*3.14)
        Al = r*r*(phi - torch.sin(phi)*torch.cos(phi))
        Hl = Al/math.pi/r**2
        Ag = (1 - Hl) * 3.14 * r**2
        Dg = 4 * Ag / (Sg + Si)
        Dl = 4 * Al / (Sl + Si)
        u_l = u_sl / Hl
        Re_g = rho_g * u_g * Dg / mu_g
        Fr = rho_g * u_g**2 / (rho_l - rho_g) / 9.81 / r / 2 / math.cos(angel)
        fg = 0.046*Re_g**(-0.2)
        fi = (1 + 5.66*(Hl*Fr*Re_g**0.2)**0.25) * fg
        tau_i = (fi*rho_g*(u_g - u_l)*abs(u_g - u_l)/2)
        tau_wg = fg*rho_g*u_g**2/2
        # print(tau_i)
        tau_wl = (tau_wg*(Sg/Ag + tau_i/tau_wg*(Si/Al + Si/Ag)) - (rho_l - rho_g)*9.8*torch.sin(angel))*Al/Sl
        error = tau_i*Si + Al*rho_l*9.8*torch.sin(angel)
        return torch.abs(tau_wl), torch.abs(error)
     
    # 最小压降模型
    def loss_pressure(self, phi, u_g, mu_l, rho_l, rho_g, mu_g, u_sl, angel, r):
        Sg = 2*3.14*r * (2*3.14 - 2*phi)/(2*3.14)
        Si = 2 * torch.sin(phi) * r
        Sl = 2*3.14*r * 2*phi/(2*3.14)
        Al = r*r*(phi - torch.sin(phi)*torch.cos(phi))
        Hl = Al/math.pi/r**2
        Ag = (1 - Hl) * 3.14 * r**2
        Dg = 4 * Ag / (Sg + Si)
        Dl = 4 * Al / (Sl + Si)
        u_l = u_sl / Hl
        Re_g = rho_g * u_g * Dg / mu_g
        Fr = rho_g * u_g**2 / (rho_l - rho_g) / 9.81 / r / 2 / math.cos(angel)
        fg = 0.046*Re_g**(-0.2)
        fi = (1 + 5.66*(Hl*Fr*Re_g**0.2)**0.25) * fg
        tau_i = (fi*rho_g*(u_g - u_l)*abs(u_g - u_l)/2)
        tau_wg = fg*rho_g*u_g**2/2
        delta = phi/math.pi
        an = torch.tensor([-0.000001, 0.000355, 6.7477988, -7.0926783, -11.959489, 19.603771, -7.0282088])
        bn = torch.tensor([1, -1.1126440, 1.5839297, -4.4506182, 5.9410713, -3.6313889, 0.9411974])
        cn = torch.tensor([0.16668, -0.164783, 1.8630015, -4.2176912, 2.352805])
        dn = torch.tensor([1, -0.9881024, 6.9011371, -21.5540619, 30.3251058, -26.9186991, 15.2478743, -3.6939891])
        sum_an = torch.tensor(0)
        sum_bn = torch.tensor(0)
        sum_cn = torch.tensor(0)
        sum_dn = torch.tensor(0)
        for i in range(7):
            if i <= 4:
                sum_an = sum_an + an[i]*delta**i
                sum_bn = sum_bn + bn[i]*delta**i
                sum_cn = sum_cn + cn[i]*delta**i
                sum_dn = sum_dn + dn[i]*delta**i
            elif i <= 6:
                sum_an = sum_an + an[i]*delta**i
                sum_bn = sum_bn + bn[i]*delta**i
                sum_dn = sum_dn + dn[i]*delta**i
            else:
                sum_dn = sum_dn + dn[i]*delta**i
        dl = sum_an/sum_bn*2*r
        fl = sum_cn/sum_dn
        tau_wl = 8*mu_l*u_l/dl - fl*tau_i
        pressure = (tau_i*Si - tau_wl*Sl - Al*rho_l*9.8*torch.sin(angel))/Al
        return torch.abs(pressure)
    
    def calc(self, mu_l, mu_g, rho_l, rho_g, u_sl, angel, r, method='inter', epoch=5000):
        x = np.array([mu_l, mu_g, rho_l, rho_g, u_sl, angel, r])
        x = torch.tensor(x.reshape(1, len(x)), dtype=torch.float32)
        self.x_min = x.min(dim=1, keepdim=True)[0]
        self.x_max = x.max(dim=1, keepdim=True)[0]
        x_ = (x - self.x_min) / (self.x_max - self.x_min)
        mu_l = torch.tensor(mu_l, dtype=torch.float32) 
        mu_g = torch.tensor(mu_g, dtype=torch.float32)
        u_sl = torch.tensor(u_sl, dtype=torch.float32)
        angel = torch.tensor(angel, dtype=torch.float32)
        rho_l = torch.tensor(rho_l, dtype=torch.float32)
        rho_g = torch.tensor(rho_g, dtype=torch.float32)
        r = torch.tensor(r, dtype=torch.float32)
        
        if method == 'wall':
            optimizer = torch.optim.RMSprop([{'params': self.phi_linears.parameters(), 'lr': 5e-4},
                                          {'params': self.ug_linears.parameters(), 'lr': 5e-3}])
            
            for i in range(epoch):
                phi ,u_g = self(x_)
                optimizer.zero_grad()
                # 零壁面剪切应力模型
                tau_wl,  error=  self.loss_wall(phi, u_g, mu_l, rho_l, rho_g, mu_g, u_sl, angel, r)
                loss = tau_wl
                loss.backward()
                optimizer.step()
                Al = r*r*(phi - torch.sin(phi)*torch.cos(phi))
                Hl = Al/math.pi/r**2
                if loss < 1e-5:
                    break
                if i % 100 == 0:
                    print(f"PGNN模型训练第{i}步：loss = {loss.item()}")
                    print(f"气相流速为{u_g.item()}步, 持液率为{Hl.item()}")
                    
        elif method == 'inter':
            optimizer = torch.optim.RMSprop([{'params': self.phi_linears.parameters(), 'lr': 8e-5},
                                          {'params': self.ug_linears.parameters(), 'lr': 5e-3}])
            
            for i in range(epoch):
                phi ,u_g = self(x_)
                optimizer.zero_grad()
                # 最小界面剪切应力模型
                loss_tau, loss_error = self.loss_inter_tau(phi, u_g, mu_l, rho_l, rho_g, mu_g, u_sl, angel, r)
                loss = loss_tau + loss_error*100
                Al = r*r*(phi - torch.sin(phi)*torch.cos(phi))
                Hl = Al/math.pi/r**2
                loss.backward()
                optimizer.step()
                if i % 100 == 0:
                    print(f"PGNN模型训练第{i}步：loss = {loss.item()}")
                    print(f"气相流速为{u_g.item()}步, 持液率为{Hl.item()}")
                    
        elif method == 'pressure':
            optimizer = torch.optim.RMSprop([{'params': self.phi_linears.parameters(), 'lr': 1e-4},
                                          {'params': self.ug_linears.parameters(), 'lr': 1e-3}])
            
            for i in range(epoch):
                phi ,u_g = self(x_)
                optimizer.zero_grad()
                loss = self.loss_pressure(phi, u_g, mu_l, rho_l, rho_g, mu_g, u_sl, angel, r)
                Al = r*r*(phi - torch.sin(phi)*torch.cos(phi))
                Hl = Al/math.pi/r**2
                loss.backward()
                optimizer.step()
                if i % 100 == 0:
                    print(f"PGNN模型训练第{i}步：loss = {loss.item()}")
                    print(f"气相流速为{u_g.item()}步, 持液率为{Hl.item()}")
        
        return Hl, u_g
    
#%% 模型测试
mu_l = 1/1000
mu_g = 0.015/1000
u_sl = 0.01
angel = 1 / 180 * 3.14
rho_l = 1000
rho_g = 1.18
r = 152.4 / 1000 / 2
model = PGNN([7, 15, 15, 1])
Hl, u_g = model.calc(mu_l, mu_g, rho_l, rho_g, u_sl, angel, r, 'pressure', 5000)
print('临界持液率为{},临界携液气速为{}'.format(Hl, u_g*(1-Hl)))
#%% 公开实验数据测试
# import pandas as pd
# Hcl_list = np.array([])
# vcg_list = np.array([])
# data = pd.read_excel(r'C:\Users\ZhangXR_CUP\Desktop\临界携液气速数据\数据整理.xlsx')
# for item in data.index.to_list():
#     print('--------第{}个数据-----------'.format(item))
#     mu_l = data['液相粘度'][item]/1000
#     mu_g = data['气相粘度'][item]/1000
#     u_sl = data['液相表观流速'][item]
#     angel = data['角度'][item]/180*3.14
#     rho_l = data['液相密度'][item]
#     rho_g = data['气相密度'][item]
#     r = data['管径'][item] / 1000 / 2
#     model = PGNN([7, 15, 15, 1])
#     Hl, u_g = model.calc(mu_l, mu_g, rho_l, rho_g, u_sl, angel, r, 'wall', 5000)
#     Hcl_list = np.append(Hcl_list, Hl.detach().numpy())
#     vcg = u_g*(1-Hl)
#     vcg_list = np.append(vcg_list, vcg.detach().numpy())
