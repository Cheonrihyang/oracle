package com.spring.view2.board;

import java.util.List;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.HttpSession;

import org.springframework.web.servlet.ModelAndView;
import org.springframework.web.servlet.mvc.Controller;

import com.spring.biz.board.BoardVO;
import com.spring.biz.board.impl.BoardDAO;
import com.spring.biz.board.impl.BoardServiceImpl;

public class GetBoardListController implements Controller{

	
	BoardServiceImpl service;
	public void setBoardService(BoardServiceImpl service) {
		this.service = service;
	}
	
	@Override
	public ModelAndView handleRequest(HttpServletRequest request, HttpServletResponse response) throws Exception {
		System.out.println("getBoardList.to 성공");
		
		//BoardDAO boardDAO = new BoardDAO();
		List<BoardVO> boardList = service.getBoardList();
		
		ModelAndView mav = new ModelAndView();
		mav.addObject("boardList", boardList);	//model
		mav.setViewName("board2/getBoardList");	//view

		
		return mav;
	}

}
