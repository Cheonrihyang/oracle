package com.spring.view.user;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.HttpSession;

import org.springframework.web.servlet.ModelAndView;
import org.springframework.web.servlet.mvc.Controller;

import com.spring.biz.user.UserVO;
import com.spring.biz.user.impl.UserDAO;
import com.spring.biz.user.impl.UserServiceImpl;

public class LoginController implements Controller{

	UserServiceImpl service;
	public void setUserService(UserServiceImpl service) {
		this.service = service;
	}
	
	@Override
	public ModelAndView handleRequest(HttpServletRequest request, HttpServletResponse response) throws Exception {
		System.out.println("userCheck.do 성공");
		
		String id = request.getParameter("id");
		String password = request.getParameter("password");
		
		UserVO userVO = new UserVO();
		userVO.setId(id);
		userVO.setPassword(password);
		
		//UserDAO userDAO = new UserDAO();
		UserVO user = service.checkUserLogin(userVO);
		
		
		ModelAndView mav = new ModelAndView();
		if(user != null) {
			HttpSession session = request.getSession();
			session.setAttribute("user", user);
			mav.setViewName("redirect:getBoardList.to");	//이동할 경로
		}else {
			mav.setViewName("redirect:login.jsp");
		}
		return mav;
	}

}
