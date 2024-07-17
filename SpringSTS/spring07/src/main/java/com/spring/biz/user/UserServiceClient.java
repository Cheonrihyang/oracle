package com.spring.biz.user;

import java.util.List;

import org.springframework.context.support.AbstractApplicationContext;
import org.springframework.context.support.GenericXmlApplicationContext;

import com.spring.biz.user.impl.UserService;

public class UserServiceClient {
	public static void main(String[] args) {
		
		AbstractApplicationContext factory = new GenericXmlApplicationContext("applicationContext.xml");
		UserService service = (UserService)factory.getBean("userService");
		
		
		//user 추가
		UserVO vo = (UserVO)factory.getBean("userVO");
		vo.setId("hong");
		vo.setPassword("1111");
		vo.setName("홍길동");
		vo.setRole("회원");
		
		//service.insertUser(vo);
		
		
		//user 수정
		UserVO updateVO = (UserVO)factory.getBean("userVO");
		updateVO.setName("홍길동");
		updateVO.setRole("");
		updateVO.setId("hong");
		//service.updateUser(updateVO);
		
		//user 삭제
		UserVO deleteVO = (UserVO)factory.getBean("userVO");
		deleteVO.setId("hong");
		//service.deleteUser(deleteVO);
		
		//user 조회
		UserVO searchVO = (UserVO)factory.getBean("userVO");
		searchVO.setId("admin");
		UserVO resultVO = service.getUser(searchVO);
		if(resultVO != null) {
			System.out.println("조회결과: "+resultVO.toString());
		}else {
			System.out.println("조회결과가 없습니다.");
		}
		
		//유저리스트 조회
		//List<UserVO> userList = service.getUserList();
		//for (UserVO user : userList) {
		//System.out.println(">>"+user.toString());
	}
	
}
