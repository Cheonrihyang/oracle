package com.spring.biz.board.impl;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.util.ArrayList;
import java.util.List;

import org.springframework.stereotype.Repository;

import com.spring.biz.board.BoardVO;
import com.spring.biz.common.JDBCUtil;
// @Component   >>   @Service   @Repository   @Controller


@Repository("boardDAO")
public class BoardDAO {
	private Connection conn = null;
	private PreparedStatement pstmt = null;
	private ResultSet rs = null;
	
	private final String BOARD_INSERT = 
			"insert into board(title, writer, content, fileaddress) values (?, ?, ?, ?)";
	private final String BOARD_UPDATE = 
			"update board set title=?, content=?, fileaddress=? where seq=?";
	private final String BOARD_DELETE = 
			"delete from board where seq=?";
	private final String BOARD_GET =
			"select * from board where seq=?";
	private final String BOARD_LIST =
			"select * from board order by seq desc";
	private final String BOARD_LIST_T =
			"select * from board where title like ? order by seq desc";
	private final String BOARD_LIST_C =
			"select * from board where content like ? order by seq desc";
	
	public void insertBoard(BoardVO vo) {
		try {
			conn = JDBCUtil.getConnection();
			pstmt = conn.prepareStatement(BOARD_INSERT);
			//insert into board(title, writer, content) values (?, ?, ?)
			pstmt.setString(1, vo.getTitle());
			pstmt.setString(2, vo.getWriter());
			pstmt.setString(3, vo.getContent());
			pstmt.setString(4, vo.getFileAddress());
			pstmt.executeUpdate();
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			JDBCUtil.close(pstmt, conn);
		}
	}
	
	public void updateBoard(BoardVO vo) {
		try {
			conn = JDBCUtil.getConnection();
			pstmt = conn.prepareStatement(BOARD_UPDATE);
			//update board set title=?, content=? where seq=?
			pstmt.setString(1, vo.getTitle());
			pstmt.setString(2, vo.getContent());
			pstmt.setInt(3, vo.getSeq());
			pstmt.setString(4, vo.getFileAddress());
			pstmt.executeUpdate();
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			JDBCUtil.close(pstmt, conn);
		}
	}
	
	public void deleteBoard(BoardVO vo) {
		try {
			conn = JDBCUtil.getConnection();
			pstmt = conn.prepareStatement(BOARD_DELETE);
			//delete from board where seq=?
			pstmt.setInt(1, vo.getSeq());
			pstmt.executeUpdate();
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			JDBCUtil.close(pstmt, conn);
		}
	}
	
	public BoardVO getBoard(BoardVO vo) {
		BoardVO board = null;
		try {
			conn = JDBCUtil.getConnection();
			pstmt = conn.prepareStatement(BOARD_GET);
			//select * from board where seq=?
			pstmt.setInt(1, vo.getSeq());
			rs = pstmt.executeQuery();
			if(rs.next()) {
				board = new BoardVO();
				board.setSeq(rs.getInt("seq"));
				board.setTitle(rs.getString("title"));
				board.setWriter(rs.getString("writer"));
				board.setContent(rs.getString("content"));
				board.setRegdate(rs.getDate("regdate"));
				board.setCnt(rs.getInt("cnt"));
				board.setFileAddress(rs.getString("fileaddress"));
			}
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			JDBCUtil.close(rs, pstmt, conn);
		}
		return board;
	}
	
	public List<BoardVO> getBoardList(BoardVO vo){
		List<BoardVO> boardList = new ArrayList<BoardVO>();
		try {
			conn = JDBCUtil.getConnection();
			if(vo.getSearchCondition().equals("TITLE") && vo.getSearchKeyword().equals("")
					|| vo.getSearchCondition().equals("CONTENT") && vo.getSearchKeyword().equals("")) {
				pstmt = conn.prepareStatement(BOARD_LIST);
			}else {
				if(vo.getSearchCondition().equals("TITLE")) {
					pstmt = conn.prepareStatement(BOARD_LIST_T);
				}else if(vo.getSearchCondition().equals("CONTENT")) {
					pstmt = conn.prepareStatement(BOARD_LIST_C);
				}
				
				pstmt.setString(1, vo.getSearchKeyword());
				
				
			}
			
			rs = pstmt.executeQuery();
			while(rs.next()) {
				BoardVO board = new BoardVO();
				board.setSeq(rs.getInt("seq"));
				board.setTitle(rs.getString("title"));
				board.setWriter(rs.getString("writer"));
				board.setContent(rs.getString("content"));
				board.setRegdate(rs.getDate("regdate"));
				board.setCnt(rs.getInt("cnt"));
				boardList.add(board);
			}
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			JDBCUtil.close(rs, pstmt, conn);
		}
		return boardList;
	}
}
