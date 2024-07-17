package com.spring.view.user;

import java.util.Calendar;

import javax.servlet.http.HttpSession;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;

import com.spring.biz.board.BoardVO;
import com.spring.biz.user.UserVO;
import com.spring.biz.user.impl.UserDAO;

@Controller
public class UserController {
	
	@RequestMapping(value="/userCheck.do" , method = RequestMethod.GET)
	public String login(UserVO vo,BoardVO bo) {
		vo.setId("admin");
		vo.setPassword("1234");
		bo.setTitle("제목");
		
		return "redirect:login.jsp";
	}
	
	@RequestMapping(value="/userCheck.do" , method = RequestMethod.POST)
	public String login(UserVO vo,UserDAO userDAO,HttpSession session) throws IllegalAccessException {
		if(vo.getId()==null || vo.getId()=="") {
			throw new IllegalAccessException("아이디는 반드시 입력하셔야 합니다.");
		}
		/*
		int num1=100;
		int num2=0;
		int sum = num1/num2;
		if(sum>0) {
			throw new IllegalArgumentException("0으로 값을 나눌수 없습니다.");
		}
		*/
		Calendar now = Calendar.getInstance();
		Object now2 = now;
		now2=null;
		Object now3 = now2;
		System.out.println(now3.toString());
		
		System.out.println("userCheck.do 성공");
		if(userDAO.checkUserLogin(vo) != null) {
			session.setAttribute("USER",userDAO.getUser(vo).getName());
			
			return "redirect:getBoardList.do";	//이동할 경로
		}else {
			return "redirect:login.jsp";
		}
	}
	
	@RequestMapping("/logout.do")
	public String logout(HttpSession session) {
		session.invalidate();
		return "redirect:login.jsp";
		
	}
}
